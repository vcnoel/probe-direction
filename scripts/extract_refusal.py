"""Extract refusal activations under the SAME 18 wrappers used for evaluation awareness.

The point of a second concept is to separate a property of the instrument from a property of the
concept, and that only works if the wrapper set is held fixed. So no refusal-specific framings are
authored: the same 18 wrappers carry harmful-versus-harmless instruction content instead of
eval-versus-deployment context.

The two concepts do not load the wrapper facet identically, and refusal's version is the cleaner
one. For evaluation awareness a template IS the contrast -- an (eval, deploy) pair -- so wrapper
variance and treatment are entangled by construction. Here the contrast is content and the wrapper
is context held fixed across both arms, so only the wrapper varies, which isolates whether the
direction depends on surrounding context.

Stored per model:
  dir_harmful/dir_harmless  [18, L+1, d]  wrapper-conditional means for the direction
  score_pooled              [400, L+1, d] held-out scoring items, fixed rendering
  score_by_wrapper          [18, 400, L+1, d] is NOT stored -- prohibitive; the transfer matrix
                            is computed in a separate ablation pass that needs logits, not pools
  last_tok_*                same as above but at the final prompt position, because the reference
                            refusal pipeline selects jointly over layer AND post-instruction token
                            position, and position is an implementation level here

    python scripts/extract_refusal.py --model gemma2-2b
    python scripts/extract_refusal.py --all
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np

from external_paths import reanalysis  # noqa: E402
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_awareness_spectral.prompts import load_framings  # noqa: E402
from extract_model import PINNED_DATE, cfg  # noqa: E402

ARDITI = reanalysis("arditi", "dataset", "splits")
N_DIR = 48        # per arm, per wrapper, for building the direction (T3.4)
N_SCORE = 200     # per arm, held out, for scoring (T3.4) -> 400 items, matching the SAD count
OUT = ROOT / "results" / "refusal_v1"


def instructions():
    rng = random.Random(0)
    def load(name):
        return [x["instruction"] for x in
                json.loads((ARDITI / f"{name}.json").read_text(encoding="utf-8"))]
    hf_tr, hl_tr = load("harmful_train"), load("harmless_train")
    hf_te, hl_te = load("harmful_test"), load("harmless_test")
    return {
        "dir_harmful": rng.sample(hf_tr, N_DIR),
        "dir_harmless": rng.sample(hl_tr, N_DIR),
        "score_harmful": rng.sample(hf_te, N_SCORE),
        "score_harmless": rng.sample(hl_te, N_SCORE),
    }


def render(tok, framing, instruction, system_role):
    """Apply one wrapper to one instruction, honouring the family's system-role convention."""
    sys_txt = framing.system
    user_txt = framing.user_template.replace("{task}", instruction)
    if system_role:
        msgs = [{"role": "system", "content": sys_txt},
                {"role": "user", "content": user_txt}]
    else:
        msgs = [{"role": "user", "content": f"{sys_txt}\n\n{user_txt}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       date_string=PINNED_DATE)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def pool_mean_and_last(model, tok, texts, log_every=200):
    """-> (mean over kept tokens, final-position) each [n, L+1, d].

    Both are returned because the reference refusal pipeline searches over token position as well
    as layer, so position is a level of the implementation facet rather than a fixed choice.
    """
    specials = {"<|begin_of_text|>", "<s>", "<BOS>", "<bos>"}
    means, lasts = [], []
    for i, t in enumerate(texts):
        enc = tok(t, return_tensors="pt").to(model.device)
        ids = enc["input_ids"][0]
        hs = model(**enc, output_hidden_states=True).hidden_states
        stack = torch.stack([h[0] for h in hs])                    # [L+1, T, d]
        keep = torch.tensor([tok.decode([x]).strip() != ""
                             and tok.decode([x]).strip() not in specials
                             for x in ids.tolist()], device=stack.device)
        if not keep.any():
            keep = torch.ones_like(keep)
        means.append(stack[:, keep, :].mean(1).float().cpu().numpy())
        lasts.append(stack[:, -1, :].float().cpu().numpy())
        del hs, stack
        if (i + 1) % log_every == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)
    return np.stack(means), np.stack(lasts)


def run(key, conf, dtype="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = conf["models"][key]
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pooled.npz"
    if dest.exists():
        print(f"{key}: already extracted"); return

    fr = load_framings(str(ROOT / "data" / "framings_multi.json"))
    wrappers = [k for k in fr if not k.startswith("_")]
    ins = instructions()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=getattr(torch, dtype), attn_implementation="eager"
    ).to("cuda").eval()
    sr = spec["system_role"]

    # direction side: every wrapper x both arms
    texts, index = [], []
    for w in wrappers:
        for arm in ("dir_harmful", "dir_harmless"):
            for instr in ins[arm]:
                texts.append(render(tok, fr[w], instr, sr))
                index.append((w, arm))
    print(f"[{key}] direction prompts: {len(texts)} "
          f"({len(wrappers)} wrappers x 2 arms x {N_DIR})", flush=True)
    dmean, dlast = pool_mean_and_last(model, tok, texts)

    # scoring side: fixed rendering, so the direction is the only thing that varies
    score_texts, score_y = [], []
    for arm, lab in (("score_harmful", 1), ("score_harmless", 0)):
        for instr in ins[arm]:
            # Bare instruction with no system turn, identical across families, so the direction is
            # the only thing that varies on the scoring side.
            msgs = [{"role": "user", "content": instr}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            date_string=PINNED_DATE)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            score_texts.append(t)
            score_y.append(lab)
    print(f"[{key}] scoring prompts: {len(score_texts)}", flush=True)
    smean, slast = pool_mean_and_last(model, tok, score_texts)

    tmp = out_dir / "pooled_partial.npz"
    np.savez_compressed(
        tmp,
        dir_mean=dmean, dir_last=dlast,
        dir_wrapper=np.array([w for w, _ in index]),
        dir_arm=np.array([a for _, a in index]),
        score_mean=smean, score_last=slast,
        score_label=np.array(score_y),
        wrappers=np.array(wrappers),
        families=json.dumps({k: v.get("family", "?") for k, v in
                             json.loads((ROOT / "data" / "framings_multi.json")
                                        .read_text(encoding="utf-8")).items()
                             if not k.startswith("_")}),
        meta=json.dumps({"key": key, "hf_id": spec["hf_id"], "family": spec["family"],
                         "params_b": spec["params_b"], "system_role": sr, "dtype": dtype,
                         "n_dir_per_arm": N_DIR, "n_score_per_arm": N_SCORE}),
    )
    tmp.replace(dest)
    print(f"[{key}] saved -> {dest}  dir={dmean.shape} score={smean.shape}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    conf = cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(conf["models"]))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    todo = ([k for k, v in conf["models"].items() if v.get("enabled")] if a.all
            else [a.model] if a.model else None)
    if not todo:
        ap.error("pass --model or --all")
    print(f"to extract: {', '.join(todo)}\n")
    for k in todo:
        run(k, conf, a.dtype)


if __name__ == "__main__":
    main()
