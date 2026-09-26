"""Is the SIGN of the reported scaling relationship determined by the template drawn?

Every published scaling claim in this literature fits the probe statistic against parameter count
using one contrastive template. This runs that same fit once per template, over the 36 matched
eval x deployment pairs, and reports the distribution of the 36 slopes.

If that distribution straddles zero, the conclusion is not that the field's estimate is noisy but
that its direction is unidentified: a study drawing one template reports a positive relationship
or a negative one depending on which it drew, and nothing in its design reveals which happened.

Two published values are placed in the distribution for reference. Both are computed by us from
released artifacts over the same four Qwen checkpoints, the same held-out split, and the same
statistic, so the comparison is like-for-like rather than across populations.

    python scripts/slope_distribution.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_analyze import load, unit  # noqa: E402
from gstudy_gtheory import auc_cols, cell_statistic  # noqa: E402

# Which SAD pooling convention a run uses. "lf" is the validated one: BOS and
# whitespace-only tokens dropped, LF newlines, no truncation. See T1.
SADFILE = {"filtered": "sad_filtered.npz", "lf": "sad_filtered_lf.npz"}

MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
# our reanalyses of the released artifacts, same four models, same split, same statistic
PUBLISHED = {
    "Chaudhary et al., as released": 0.733,
    "Chaudhary et al., 3B/7B un-transposed": 0.975,
    "Manek et al.": -0.089,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistic", default="peak", choices=["peak", "tilt"])
    ap.add_argument("--facet", default="deploy", choices=["deploy", "neutral"])
    ap.add_argument("--sad-pooling", default="lf", choices=["all", "filtered", "lf"])
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    loaded = {m: d for m in MODELS if (d := load(m)) is not None}
    if a.sad_pooling != "all":
        for m, d in list(loaded.items()):
            p = ROOT / "results" / "gstudy" / m / SADFILE[a.sad_pooling]
            with np.load(p, allow_pickle=False) as z:
                d["sad"] = z["sad"].astype(np.float64); d["y"] = z["sad_label"].astype(int)
    models = list(loaded)
    fam = next(iter(loaded.values()))["fam"]
    evals = sorted(k for k, v in fam.items() if v == "eval")
    others = sorted(k for k, v in fam.items() if v == a.facet)
    pairs = list(itertools.product(evals, others))

    stat = np.zeros((len(models), len(pairs)))
    for i, m in enumerate(models):
        d = loaded[m]
        names = np.array(d["fname"])
        for j, (fe, fo) in enumerate(pairs):
            direction = unit(d["fr"][names == fe].mean(axis=0) - d["fr"][names == fo].mean(axis=0))
            proj = np.einsum("ild,ld->il", d["sad"], direction)
            stat[i, j] = cell_statistic(d["y"], proj, a.statistic)

    x = np.log10([PARAMS[m] for m in models])
    xc = x - x.mean()
    slopes = np.array([float((xc * (stat[:, j] - stat[:, j].mean())).sum() / (xc ** 2).sum())
                       for j in range(len(pairs))])
    rs = np.array([float(np.corrcoef(x, stat[:, j])[0, 1]) for j in range(len(pairs))])

    pos, neg = int((slopes > 0).sum()), int((slopes < 0).sum())
    print(f"{'=' * 92}\nSLOPE OF THE STATISTIC ON log10(PARAMETERS), FITTED ONCE PER TEMPLATE"
          f"\n{'=' * 92}")
    print(f"  {len(pairs)} templates x {len(models)} models ({a.facet} facet, {a.statistic})\n")
    print(f"  slope   min {slopes.min():+.4f}   median {np.median(slopes):+.4f}   "
          f"max {slopes.max():+.4f}")
    print(f"  sign    POSITIVE in {pos}/{len(pairs)} templates ({100 * pos / len(pairs):.0f}%),  "
          f"NEGATIVE in {neg}/{len(pairs)} ({100 * neg / len(pairs):.0f}%)")
    print(f"  correlation r  min {rs.min():+.3f}   median {np.median(rs):+.3f}   "
          f"max {rs.max():+.3f}")
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    print(f"  central 95% of template slopes: [{lo:+.4f}, {hi:+.4f}]")

    print(f"\n  where the published values fall in the distribution of template correlations:")
    for label, r in PUBLISHED.items():
        pct = 100.0 * float((rs <= r).mean())
        inside = "inside" if rs.min() <= r <= rs.max() else "OUTSIDE the range we observe"
        print(f"    {label:<40} r={r:+.3f}   {pct:5.1f}th percentile   {inside}")

    print(f"\n  histogram of the {len(pairs)} template correlations")
    edges = np.linspace(-1, 1, 21)
    counts, _ = np.histogram(rs, bins=edges)
    for c, lo_, hi_ in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c)
        mark = ""
        for label, r in PUBLISHED.items():
            if lo_ <= r < hi_:
                mark += f"   <- {label}"
        print(f"    [{lo_:+.1f},{hi_:+.1f})  {bar:<20}{c:3d}{mark}")

    verdict = ("the sign is NOT identified: both directions occur across templates, so a "
               "single-template study reports whichever it drew"
               if pos and neg else
               "every template agrees in sign; the direction is identified even if the "
               "magnitude is not")
    print(f"\n  VERDICT  {verdict}")

    out = {"statistic": a.statistic, "facet": a.facet, "n_templates": len(pairs),
           "models": models, "slopes": slopes.tolist(), "correlations": rs.tolist(),
           "frac_positive": pos / len(pairs), "frac_negative": neg / len(pairs),
           "slope_ci95": [float(lo), float(hi)], "published": PUBLISHED}
    suffix = f"_{a.sad_pooling}"
    p = ROOT / "results" / f"slope_distribution_{a.facet}_{a.statistic}{suffix}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
