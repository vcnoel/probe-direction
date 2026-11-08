"""Pooled activations of the original and paraphrased framings (docs/paraphrase-registration.md).

Qwen3 and Qwen3.5 (amendment 1), non-thinking rendering.

36 framings (the 12 evaluation and deployment framings of data/framings_multi.json and their 24
paraphrases in data/framings_paraphrase.json) x the 60 tasks of data/tasks_bench.jsonl, rendered
in the paper's non-thinking Qwen3 rendering by the same calls as scripts/render_prompts.py, and
pooled by pod/extract.py's pool_reductions, the code that produced the paper's Qwen3 arrays. The
SAD scoring set is not re-extracted: the analysis reads it from the paper's stored arrays, so the
weights here are pinned to the revisions recorded there and the chat-template hash must match.

Writes <out>/<model>__nothink.npz with framing_validated and framing_alltokens (float16 storage of
bfloat16 activations, as the paper's pod arrays), framing_name, framing_task and meta. Atomic and
resumable: a finished model is skipped.

    EAS_ACTIVATION_ROOT=<root> python scripts/extract_paraphrase.py --model qwen3-0.6b
    python scripts/extract_paraphrase.py --render-only        # render and hash, no model weights
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from eval_awareness_spectral.prompts import build_prompts, load_framings, load_tasks  # noqa: E402

_spec = importlib.util.spec_from_file_location("pod_extract", ROOT / "pod" / "extract.py")
pod_extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod_extract)

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
OUT = ACT / "results" / "post_submission" / "paraphrase" / "acts"
PINNED_DATE = "01 Jan 2025"                       # scripts/render_prompts.py
VARIANT, TEMPLATE_KWARGS = "nothink", {"enable_thinking": False}
# repo, revision recorded in results/pod_qwen3[5]/<model>__nothink.npz meta, bf16 bytes of the
# weights that reach the GPU (Qwen3.5: text decoder only), the paper's chat-template hash
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 1503300328, "a55ee1b16601"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 4063479808, "a55ee1b16601"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 8044936192, "a55ee1b16601"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", 16381470720, "a55ee1b16601"),
    "qwen3-14b": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", 29536614400, "a55ee1b16601"),
    "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", 65524246528, "a55ee1b16601"),
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", 1504791232, "273d8e0e683b"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", 3763655360, "273d8e0e683b"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", 8411510272, "a4aee8afcf2e"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 17907614208, "a4aee8afcf2e"),
    "qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", 53792013824, "a4aee8afcf2e"),
}


def git_state():
    """Commit and working-tree state of this checkout, recorded in every extraction's meta."""
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as e:                           # no git on the host: say so
            return f"unavailable: {e}"
    return {"git_commit": run("rev-parse", "HEAD"), "git_status_porcelain": run("status", "--porcelain")}


def framings():
    """The 12 originals (file order) followed by the 24 paraphrases (file order)."""
    multi = json.loads((ROOT / "data" / "framings_multi.json").read_text(encoding="utf-8"))
    orig = [k for k, v in multi.items() if not k.startswith("_") and v.get("family") in ("eval", "deploy")]
    conds = {k: v for k, v in load_framings(str(ROOT / "data" / "framings_multi.json")).items()
             if k in orig}
    para = load_framings(str(ROOT / "data" / "framings_paraphrase.json"))
    assert set(para) == {f"{k}_p{v}" for k in orig for v in (1, 2)}, "paraphrase key set"
    conds.update(para)
    return conds


def render(tok, want_sha12):
    th = hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()[:12]
    if th != want_sha12:
        raise SystemExit(f"chat template {th} is not the paper's {want_sha12}")
    conds = framings()
    tasks = load_tasks(str(ROOT / "data" / "tasks_bench.jsonl"))
    prompts = build_prompts(tasks, conds, tok, only_conditions=list(conds), position_match=False,
                            template_kwargs=dict(TEMPLATE_KWARGS, date_string=PINNED_DATE))
    texts = [p.text.replace("\r\n", "\n") for p in prompts]
    for p, t in zip(prompts, texts):
        if "\r" in t:
            raise SystemExit(f"CR survived rendering in {p.condition}/{p.task_id}")
        if p.task_text not in t:
            raise SystemExit(f"task text not verbatim in {p.condition}/{p.task_id}")
    rows = [{"name": p.condition, "task": p.task_id, "text": t} for p, t in zip(prompts, texts)]
    sha = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False)
                         .encode("utf-8")).hexdigest()
    return rows, sha, th


def run(key, batch, delete_weights=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev, wbytes, want = MODELS[key]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{key}__{VARIANT}.npz"
    if dest.exists():
        pod_extract.log(f"{dest.name}: already extracted, skipping")
        return None
    if batch <= 0:                      # the paper's pod rule (pod/extract.py batch_for)
        batch = pod_extract.batch_for(wbytes / 1e9, pod_extract.vram_total())
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    rows, sha, th = render(tok, want)
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda").eval()
    t_load = time.time() - t0
    pod_extract.log(f"{key}: loaded in {t_load:.0f}s; {len(rows)} prompts, batch {batch}, "
                    f"prompt sha {sha[:12]}")
    t1 = time.time()
    out = pod_extract.pool_reductions(model, tok, [r["text"] for r in rows], batch)
    t_fwd = time.time() - t1
    meta = {"key": f"{key}__{VARIANT}", "model_key": key, "repo": repo, "revision": rev,
            "variant": VARIANT, "template_kwargs": TEMPLATE_KWARGS, "template_sha12": th,
            "pinned_date": PINNED_DATE, "prompt_sha256": sha, "n_prompts": len(rows),
            "framings": sorted({r["name"] for r in rows}), "batch": batch,
            "dtype": "bfloat16", "storage_dtype": "float16", "attn": "sdpa",
            "pooling_filter": "BOS and whitespace-only tokens dropped for 'validated'",
            "seconds_load": t_load, "seconds_forward": t_fwd,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, **git_state()}
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, framing_validated=out["validated"],
                        framing_alltokens=out["alltokens"],
                        framing_name=np.array([r["name"] for r in rows]),
                        framing_task=np.array([r["task"] for r in rows]),
                        meta=np.array(json.dumps(meta)))
    with np.load(tmp, allow_pickle=False) as z:
        assert z["framing_validated"].shape[0] == len(rows)
        assert np.isfinite(z["framing_validated"]).all(), "non-finite pooled activation"
    tmp.replace(dest)
    pod_extract.log(f"{dest.name}: {dest.stat().st_size / 1e6:.0f}MB, forward {t_fwd:.0f}s, "
                    f"total {time.time() - t0:.0f}s")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if delete_weights:
        snap = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        for d in snap.glob(f"models--{repo.replace('/', '--')}*"):
            shutil.rmtree(d, ignore_errors=True)
            pod_extract.log(f"{key}: deleted weights {d.name}")
    return {"key": key, "seconds_load": t_load, "seconds_forward": t_fwd,
            "seconds_total": time.time() - t0, "weight_gb": wbytes / 1e9}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, default=16,
                    help="0: the paper's pod rule, min(16, VRAM headroom / 1.5 GB)")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--delete-weights", action="store_true",
                    help="pod: remove the snapshot after a model is written")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if a.render_only:
        from transformers import AutoTokenizer
        for key in a.model or ["qwen3-0.6b"]:
            repo, rev, _, want = MODELS[key]
            rows, sha, th = render(AutoTokenizer.from_pretrained(repo, revision=rev), want)
            print(f"{key}: {len(rows)} prompts, {len({r['name'] for r in rows})} framings, "
                  f"template {th}, sha {sha[:12]}")
        return
    if not a.model:
        ap.error("pass --model (repeatable)")
    for key in a.model:                 # one model in memory at a time
        r = run(key, a.batch, a.delete_weights)
        if r:
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
