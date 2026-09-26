"""Extract mean-pooled activations for a generalizability study of probe measurement.

Every published evaluation-awareness probe result is conditional on ONE contrastive template
pair, chosen once and never varied. This extracts the two ingredients needed to measure how
much the field's headline statistic moves when that fixed choice is varied:

  A. the held-out evaluation set actually used in the literature -- the 400-item SAD
     stages_oversight split, taken verbatim (already chat-templated) from the released
     probe-result CSVs, so scores are comparable to published numbers;
  B. our 13 framings x 60 tasks, from which many different contrastive directions can be
     built (6 eval framings x 7 non-eval framings = 42 template pairs).

Pooling is the mean over all prompt tokens for both, which reproduces the field's per-question
`mean_score`: mean_t (h_t . d) = (mean_t h_t) . d, so a mean-pooled activation dotted with the
probe direction equals their mean-over-token projection exactly.

    python scripts/gstudy_extract.py --model qwen2.5-3b
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
from sad_source import load_sad_frame  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from eval_awareness_spectral.prompts import build_prompts, load_framings, load_tasks  # noqa: E402

LADDER = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct", "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct", "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
}
# SAD items are read from a password-protected archive outside the tree, never from a
# plaintext file in this repository. See scripts/sad_source.py.


@torch.no_grad()
def pool(model, tok, texts, max_len=1024, log_every=100):
    """-> [n_texts, n_layers+1, d] mean over non-padding tokens, float32 on CPU."""
    out = []
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states     # tuple[L+1] of [1,T,d]
        mask = enc["attention_mask"][0].bool()
        stacked = torch.stack([h[0][mask].mean(0) for h in hs])         # [L+1, d]
        out.append(stacked.float().cpu().numpy())
        if (i + 1) % log_every == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(LADDER))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=str(ROOT / "results" / "gstudy"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pooled.npz"

    # Extraction is incremental. Adding framings to the set should not force a re-run of the
    # 400-item SAD pass or of framings already pooled: activations are deterministic given
    # (model, dtype, text), so previously pooled rows are reusable verbatim. Only framings
    # absent from the existing archive are run, and the result is merged back.
    prior = None
    if dest.exists():
        with np.load(dest, allow_pickle=False) as z:      # close the handle: on Windows an open
            prior = {k: z[k] for k in z.files}            # NpzFile blocks the rename below
        have = set(str(s) for s in prior["framing_name"])
        want = {k for k in json.loads((ROOT / "data" / "framings_multi.json")
                                      .read_text(encoding="utf-8")) if not k.startswith("_")}
        missing = sorted(want - have)
        if not missing:
            print(f"{args.model} already extracted ({len(have)} framings)"); return
        print(f"[{args.model}] extending: {len(have)} framings present, "
              f"{len(missing)} to add -> {', '.join(missing)}", flush=True)

    hf_id = LADDER[args.model]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to("cuda").eval()

    # A. held-out SAD set, verbatim (already chat-templated in the release)
    sad = load_sad_frame()   # decrypted in memory; see scripts/sad_source.py
    if prior is None:
        print(f"[{args.model}] SAD stages_oversight: {len(sad)} prompts", flush=True)
        sad_vec = pool(model, tok, sad["question"].tolist())
    else:
        sad_vec = prior["sad"]
        print(f"[{args.model}] reusing pooled SAD set ({sad_vec.shape[0]} prompts)", flush=True)

    # B. our framings x tasks, same pooling -> contrastive directions live in the same space
    framings = load_framings(str(ROOT / "data" / "framings_multi.json"))
    names = [k for k in framings if not k.startswith("_")]
    tasks = load_tasks(str(ROOT / "data" / "tasks_bench.jsonl"))
    todo = missing if prior is not None else names
    prompts = build_prompts(tasks, framings, tok, only_conditions=todo, position_match=False)
    print(f"[{args.model}] framing prompts to pool: {len(prompts)}", flush=True)
    fr_vec = pool(model, tok, [p.text for p in prompts])
    fr_name = np.array([p.condition for p in prompts])
    fr_task = np.array([p.task_id for p in prompts])
    if prior is not None:
        fr_vec = np.concatenate([prior["framing"], fr_vec])
        fr_name = np.concatenate([prior["framing_name"].astype(str), fr_name])
        fr_task = np.concatenate([prior["framing_task"].astype(str), fr_task])

    # Write to a side name and rename only once complete: a reader that polls for pooled.npz
    # while savez_compressed is still streaming gets a BadZipFile, which has happened here.
    tmp = out_dir / "pooled_partial.npz"
    np.savez_compressed(
        tmp,
        sad=sad_vec, sad_label=(sad["type"] == "testing").astype(int).to_numpy(),
        sad_qid=sad["question_id"].to_numpy(),
        framing=fr_vec,
        framing_name=fr_name,
        framing_task=fr_task,
        families=json.dumps({k: v.get("family", "?") for k, v in
                             json.loads((ROOT / "data" / "framings_multi.json")
                                        .read_text(encoding="utf-8")).items()
                             if not k.startswith("_")}),
    )
    tmp.replace(dest)
    print(f"[{args.model}] saved -> {dest}  sad={sad_vec.shape} framing={fr_vec.shape} "
          f"({len(set(fr_name.tolist()))} framings)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
