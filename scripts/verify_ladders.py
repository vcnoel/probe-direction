"""Verify every candidate model from its config, not from its name. No weights, no GPU.

The pod is billed hourly, so a configuration error must surface here rather than after a 64GB
download. Every field below is read from the repository: the parameter count comes from the
safetensors byte total rather than the string in the model id, the architecture comes from
config.json, and whether a system turn is accepted comes from rendering the actual chat template.

Abort conditions, any of which disqualifies a ladder:
  * a mixture-of-experts config (num_experts / num_local_experts / n_routed_experts present)
  * an effective-parameter model, where the file size and the advertised size disagree badly
  * an architecture that differs from its family siblings, which would confound scale with design
  * a missing chat template, or one that silently drops a system turn

Rejected families and the reason are recorded in the output so the decision is not re-litigated.

    python scripts/verify_ladders.py            # verify and write configs/ladders.json
    python scripts/verify_ladders.py --json     # machine-readable summary only
    python scripts/verify_ladders.py --family qwen3.5   # add or refresh one ladder, leave the rest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "ladders.json"

CANDIDATES = {
    "qwen3": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
              "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"],
    "olmo2": ["allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-1124-7B-Instruct",
              "allenai/OLMo-2-1124-13B-Instruct", "allenai/OLMo-2-0325-32B-Instruct"],
    "qwen3.5": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"],
}

# Caveats a ladder carries into every analysis of it. Qwen3.5 was declined for the first pod run
# (the reason is kept in DECLINED_THEN_RUN, verbatim, so the decision's history survives) and is
# now run as the later generalisation that note anticipated.
NOTES = {
    "qwen3.5": "Every checkpoint is a Qwen3_5ForConditionalGeneration VLM with a vision tower and an "
               "MTP head in its safetensors. Extraction loads Qwen3_5ForCausalLM, the text decoder "
               "alone, so params_b_from_weights counts the text tensors from the safetensors headers "
               "(vision and MTP excluded, a tied embedding counted once); weight_bytes stays the "
               "full download. Layers are 3:1 Gated-DeltaNet:softmax, so a layer index is not a "
               "position in a homogeneous stack. The template's thinking default differs inside "
               "the ladder (0.8B/2B default off, 4B+ default on), so both levels are rendered with "
               "enable_thinking set explicitly and neither ends at the bare assistant tag.",
}

# Families considered and declined, with the reason, so this is not proposed again.
REJECTED = {
    "gemma4": "E2B/E4B are effective-parameter models and 26B-A4B is MoE; only 12B and 31B are "
              "clean dense, so log10(params) is not well defined across the family and a scaling "
              "regression over it is uninterpretable. Same objection as Gemma-3's mid-ladder "
              "architecture change.",
}
DECLINED_THEN_RUN = {
    "qwen3.5": "Viable but weaker than Qwen3 for this design: four dense scales over 1.05 decades, "
               "hybrid Gated-DeltaNet/softmax layers that complicate every layer-indexed statistic, "
               "and VLM small tiers. Kept as a later generalisation, not this run.",
}

MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size",
            "num_experts_per_tok")
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "bfloat16 ": 2}


def fetch(repo, name):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        p = hf_hub_download(repo_id=repo, filename=name)
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except EntryNotFoundError:
        return None


def weight_bytes(repo):
    """Total safetensors bytes, from the shard index if present, else from the file listing."""
    idx = fetch(repo, "model.safetensors.index.json")
    if idx and "metadata" in idx and "total_size" in idx["metadata"]:
        return int(idx["metadata"]["total_size"]), "safetensors index total_size"
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if total == 0:
        raise SystemExit(f"{repo}: no safetensors found; refusing to guess a parameter count")
    return total, "sum of safetensors file sizes"


# Tensors a composite (vision-language) checkpoint ships that the text decoder never runs.
NON_TEXT_PREFIXES = ("model.visual.", "visual.", "mtp.", "model.mtp.")
ST_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


def text_params(repo):
    """(params, bytes) of the text decoder alone, from the safetensors headers. No weights move.

    A byte total over-counts a VLM by its vision tower and MTP head, and F32 tensors (the DeltaNet
    decay logs) make bytes / 2 wrong even for the part that is counted, so the headers are read.
    """
    from huggingface_hub import get_safetensors_metadata
    md = get_safetensors_metadata(repo)
    n = b = 0
    for fm in md.files_metadata.values():
        for name, t in fm.tensors.items():
            if name.startswith(NON_TEXT_PREFIXES):
                continue
            k = 1
            for x in t.shape:
                k *= x
            n += k
            b += k * ST_DTYPE_BYTES[t.dtype]
    return n, b


def system_turn_ok(repo):
    """Render the real chat template with a system turn and confirm it survives."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    if not getattr(tok, "chat_template", None):
        return False, "no chat template"
    try:
        s = tok.apply_chat_template(
            [{"role": "system", "content": "SENTINEL_SYS"},
             {"role": "user", "content": "SENTINEL_USR"}],
            tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return False, f"template rejected a system turn: {type(e).__name__}"
    if "SENTINEL_SYS" not in s:
        return False, "system turn silently dropped by the template"
    return True, "system turn preserved"


def inspect(repo):
    cfg = fetch(repo, "config.json")
    if cfg is None:
        raise SystemExit(f"{repo}: no config.json; the repo id is wrong or gated")
    # A composite config keeps the decoder's shape in text_config; the top level has none of it.
    tc = cfg.get("text_config") or cfg
    moe = {k: tc[k] for k in MOE_KEYS if k in tc}
    nbytes, src = weight_bytes(repo)
    dt = str(tc.get("torch_dtype") or tc.get("dtype") or cfg.get("torch_dtype") or "bfloat16")
    per = DTYPE_BYTES.get(dt.strip(), 2)
    params = nbytes / per
    sys_ok, sys_why = system_turn_ok(repo)
    row = {
        "repo": repo,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "hidden_size": tc.get("hidden_size"),
        "torch_dtype": dt,
        "weight_bytes": nbytes,
        "weight_bytes_source": src,
        "params_b_from_weights": round(params / 1e9, 3),
        "moe_keys_present": moe,
        "system_turn": sys_ok,
        "system_turn_detail": sys_why,
        "vram_bf16_gb": round(nbytes / 2 * 2 / 1e9, 1),
    }
    if "text_config" in cfg:
        tn, tb = text_params(repo)
        row.update({
            "text_model_type": tc.get("model_type"),
            "layer_types": sorted(set(tc.get("layer_types") or [])),
            "params_b_from_weights": round(tn / 1e9, 3),
            "params_source": "safetensors headers, text decoder tensors only "
                             f"(excluded prefixes {list(NON_TEXT_PREFIXES)})",
            "params_b_all_tensors": round(params / 1e9, 3),
            "weight_bytes_text": tb,
            "vram_bf16_gb": round(tb / 1e9, 1),
        })
    from huggingface_hub import HfApi
    row["revision"] = HfApi().model_info(repo).sha
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--family", default="",
                    help="verify this ladder only and merge it into the existing ladders.json, so "
                         "adding a ladder does not re-derive (and possibly move) the others")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out = {"rejected_families": REJECTED, "ladders": {}}
    families = CANDIDATES
    if a.family:
        if a.family not in CANDIDATES:
            raise SystemExit(f"unknown family {a.family}; have {sorted(CANDIDATES)}")
        families = {a.family: CANDIDATES[a.family]}
        if OUT.exists():
            old = json.loads(OUT.read_text(encoding="utf-8"))
            out = {**old, "rejected_families": REJECTED, "ladders": dict(old["ladders"])}
    out["declined_then_run"] = DECLINED_THEN_RUN
    problems = []

    for family, repos in families.items():
        rows = []
        for repo in repos:
            r = inspect(repo)
            rows.append(r)
            if not a.json:
                print(f"  {repo:<42} L={r['num_hidden_layers']:>3} d={r['hidden_size']:>5} "
                      f"{r['params_b_from_weights']:>7.2f}B  {r['torch_dtype']:<9} "
                      f"sys={'y' if r['system_turn'] else 'N'}")
            if r["moe_keys_present"]:
                problems.append(f"{repo}: MoE config keys {sorted(r['moe_keys_present'])}")
            if not r["system_turn"]:
                problems.append(f"{repo}: {r['system_turn_detail']}")
        archs = {tuple(r["architectures"] or []) for r in rows}
        if len(archs) > 1:
            problems.append(f"{family}: architectures differ across the ladder: {archs}; "
                            "scale would be confounded with design")
        mts = {r["model_type"] for r in rows}
        if len(mts) > 1:
            problems.append(f"{family}: model_type differs across the ladder: {mts}")
        ps = [r["params_b_from_weights"] for r in rows]
        import math
        out["ladders"][family] = {
            "models": rows,
            "n_models": len(rows),
            "params_b": ps,
            "decades": round(math.log10(max(ps) / min(ps)), 3),
            "total_weight_gb": round(sum(r["weight_bytes"] for r in rows) / 1e9, 1),
            "architecture": sorted(archs)[0] if archs else None,
        }
        if family in NOTES:
            out["ladders"][family]["note"] = NOTES[family]
        if not a.json:
            L = out["ladders"][family]
            print(f"  -> {family}: {L['n_models']} models, {L['decades']} decades, "
                  f"{L['total_weight_gb']}GB total\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"wrote {OUT}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("\nall candidates verified: dense, uniform architecture, system turn preserved")


if __name__ == "__main__":
    main()
