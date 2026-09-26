"""Which half of our pipeline fails to reproduce the released numbers?

We could not reproduce the released Qwen values with any combination of prompt template, pooling
and layer set. That failure has two possible locations and they must be separated before
anything is claimed about anyone's reproducibility:

    (a) DIRECTION BUILDING -- our extraction of contrastive activations, or
    (b) SCORING -- our SAD activations and the AUROC computation on top of them.

The release ships its own saved layerwise direction vectors. Scoring THOSE with OUR SAD
activations holds (a) fixed at the authors' own output and tests (b) alone.

Two poolings of the SAD side are compared, because reading the released scoring code shows the
difference is there. `analyze_dataset_with_probe` projects every token and then discards BOS and
any token whose decoded string strips to empty before averaging, and Qwen chat templates contain
many newline-only tokens. The identity that lets a pooled activation stand in for a
mean-over-token projection is exact only when both sides use the same token set:

    mean_t (h_t . d) = (mean_t h_t) . d.

Vector layer_i is hook_resid_pre of block i, which is hidden_states[i]. The released per-layer
files run 1..L-1, so index 0 and the final layer are excluded.

    python scripts/bisect_reproduction.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MANEK = reanalysis("manek", "notebooks")
TAGS = {"qwen2.5-0.5b": "0.5B", "qwen2.5-1.5b": "1.5B",
        "qwen2.5-3b": "3B", "qwen2.5-7b": "7B"}
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_paths import reanalysis  # noqa: E402
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
PUBLISHED = {"qwen2.5-0.5b": 0.2115, "qwen2.5-1.5b": 0.2912,
             "qwen2.5-3b": 0.2924, "qwen2.5-7b": 0.1904}


def load_vectors(tag, kind="vectors"):
    d = glob.glob(str(MANEK / f"QWEN_{tag}_CONTRASTIVE_*" / kind))
    if not d:
        return None
    out = {}
    for f in glob.glob(f"{d[0]}/layer_*.pt"):
        i = int(re.search(r"layer_(\d+)", f).group(1))
        v = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = next(iter(v.values()))
        out[i] = np.asarray(torch.as_tensor(v).float().flatten())
    return out


def best_over_layers(sad, y, vecs):
    best, best_L = 0.0, 0
    n = sad.shape[1]                                  # L+1
    for i, v in sorted(vecs.items()):
        if i < 1 or i >= n - 1:                       # released files are 1..L-1
            continue
        nv = v / max(float(np.linalg.norm(v)), 1e-12)
        a = roc_auc_score(y, sad[:, i, :] @ nv)
        d = abs(max(a, 1 - a) - 0.5)
        if d > best:
            best, best_L = d, i
    return best, best_L


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"{'=' * 98}\nBISECTION: the authors' own direction vectors, scored with our SAD "
          f"activations\n{'=' * 98}")
    print(f"  {'model':<14}{'SAD pooling':<14}{'ours':>10}{'published':>11}{'|dev|':>9}"
          f"{'peak layer':>12}")
    rows = {}
    for key, tag in TAGS.items():
        vecs = load_vectors(tag)
        if not vecs:
            continue
        for pooling, fname in (("all tokens", "pooled.npz"),
                               ("ws-filtered", "sad_filtered.npz"),
                               ("ws+LF+notrunc", "sad_filtered_lf.npz")):
            p = ROOT / "results" / "gstudy" / key / fname
            if not p.exists():
                continue
            with np.load(p, allow_pickle=False) as z:
                sad, y = z["sad"].astype(np.float64), z["sad_label"].astype(int)
            if vecs[0].shape[0] != sad.shape[2]:
                print(f"  {key:<14}{pooling:<14}  dim mismatch, skipped")
                continue
            b, L = best_over_layers(sad, y, vecs)
            rows.setdefault(pooling, {})[key] = b
            print(f"  {key:<14}{pooling:<14}{b:10.4f}{PUBLISHED[key]:11.4f}"
                  f"{abs(b - PUBLISHED[key]):9.4f}{L:12d}")

    spread = max(PUBLISHED.values()) - min(PUBLISHED.values())
    ms = list(TAGS)
    x = np.log10([PARAMS[m] for m in ms])
    print(f"\n  between-model spread in the published values is {spread:.4f}")
    print(f"  published correlation with log10(parameters): "
          f"{pearsonr(x, [PUBLISHED[m] for m in ms])[0]:+.3f}\n")
    for pooling, r in rows.items():
        if len(r) < len(PUBLISHED):
            print(f"  {pooling}: incomplete ({len(r)}/{len(PUBLISHED)} models)")
            continue
        dev = max(abs(r[k] - PUBLISHED[k]) for k in r)
        mean_dev = float(np.mean([abs(r[k] - PUBLISHED[k]) for k in r]))
        rr = pearsonr(x, [r[m] for m in ms])[0]
        agree = "SAME" if rr * pearsonr(x, [PUBLISHED[m] for m in ms])[0] > 0 else "OPPOSITE"
        print(f"  {pooling:<12} max|dev| {dev:.4f} ({100 * dev / spread:.0f}% of spread), "
              f"mean {mean_dev:.4f}, r={rr:+.3f}  sign {agree}")

    print(f"\n  A residual of a few hundredths is not negligible here: the whole between-model")
    print(f"  spread is {spread:.3f}, so a deviation of that size can and does reverse the sign of")
    print(f"  a four-point correlation. Until it is closed, no claim about another group's")
    print(f"  numbers is supportable, and the sign of our own scale relationship is not either.")
    (ROOT / "results" / "bisect_reproduction.json").write_text(
        json.dumps({"published": PUBLISHED, **rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {ROOT / 'results' / 'bisect_reproduction.json'}")


if __name__ == "__main__":
    main()
