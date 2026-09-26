"""Re-pool the SAD activations over the token set the released scoring code actually uses.

Our pooling averaged over every token. The released `analyze_dataset_with_probe` builds a
per-token projection and then drops BOS and every token whose decoded string strips to empty
before taking the mean. Qwen chat templates contain many newline-only tokens, so the two means
are taken over different denominators and differ systematically.

The identity that lets a pooled activation stand in for a mean-over-token projection,

    mean_t (h_t . d) = (mean_t h_t) . d,

is exact only when both sides use the same token set, so matching the field's number requires
matching its filter. This writes a second pooled tensor under the same key layout so the two can
be compared directly.

    python scripts/extract_sad_filtered.py --model qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
from sad_source import load_sad_frame  # noqa: E402
LADDER = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct", "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(LADDER))
    ap.add_argument("--dtype", default="bfloat16")
    # The prompts were recovered from released CSVs, and that round trip turned every newline
    # into CRLF. BPE tokenises "\r\n" differently from "\n", so the token boundaries -- and
    # therefore the token set the mean is taken over -- differ from the pipeline that produced
    # the reference numbers, which fed strings straight from apply_chat_template.
    ap.add_argument("--lf", action="store_true", help="restore LF newlines before tokenising")
    # The reference pipeline tokenises at the model's full context. Ours truncated at 1024,
    # which silently altered 10 of the 400 items.
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    dest = ROOT / "results" / "gstudy" / a.model / f"sad_filtered{a.suffix}.npz"
    if dest.exists():
        print(f"{a.model} already done"); return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sad = load_sad_frame()   # decrypted in memory; see scripts/sad_source.py
    tok = AutoTokenizer.from_pretrained(LADDER[a.model])
    model = AutoModelForCausalLM.from_pretrained(
        LADDER[a.model], dtype=getattr(torch, a.dtype), attn_implementation="eager"
    ).to("cuda").eval()

    out, dropped = [], []
    texts = sad["question"].tolist()
    if a.lf:
        texts = [t.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
                 for t in texts]
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=a.max_len).to(model.device)
        ids = enc["input_ids"][0]
        keep = torch.tensor(
            [tok.decode([t]).strip() != "" and tok.decode([t]).strip() not in
             ("<|begin_of_text|>", "<s>", "<BOS>") for t in ids.tolist()],
            device=ids.device)
        if not keep.any():
            keep = torch.ones_like(keep)
        dropped.append(float((~keep).float().mean()))
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0][keep].mean(0) for h in hs]).float().cpu().numpy())
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)

    arr = np.stack(out)
    np.savez_compressed(dest, sad=arr,
                        sad_label=(sad["type"] == "testing").astype(int).to_numpy())
    print(f"[{a.model}] saved {arr.shape} -> {dest}   mean fraction of tokens dropped "
          f"{100 * np.mean(dropped):.1f}%", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
