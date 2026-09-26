"""Does the reported evaluation-awareness scaling law survive selection correction?

The released per-model statistic is max over layers of |AUROC - 0.5| (the summary CSV column
is labelled "AUROC" but equals the absolute distance from chance; e.g. Qwen2.5-3B is listed at
0.2887 while its layer-27 AUROC is 0.7887). Two consequences:

* folding to |AUROC - 0.5| makes every layer's noise contribute POSITIVELY, so a maximum over
  layers is upward biased even under a pure null;
* the number of candidate layers grows with model size (17 -> 80 here), so the size of that
  bias grows with model size, which can manufacture a positive size-vs-awareness relationship
  out of noise alone.

This script quantifies the bias and refits the relationship after subtracting it. Under H0,
AUROC is approximately Normal(0.5, sigma) with sigma = sqrt((n1+n2+1)/(12 n1 n2)); the expected
maximum of |.| over L_eff effectively independent layers is approximately
sigma * sqrt(2 ln(2 L_eff)). Adjacent layers are correlated, so L_eff is estimated from the
lag-1 autocorrelation of each model's own layer curve rather than assumed equal to L.

This is a sensitivity analysis on released summary data, not a claim that the measurements are
wrong: the per-layer AUROC values are taken exactly as published. No item-level scores are
released, so no confidence interval on any individual AUROC is possible from these artifacts.

    python scripts/reanalyze_scaling_law.py --dir ../eval-awareness-reanalysis/chaud
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# file stem -> (display name, parameters in billions)
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
def null_max(sigma: float, l_eff: float) -> float:
    """Expected max of |N(0, sigma)| over l_eff effectively independent draws."""
    l_eff = max(l_eff, 1.0)
    return float(sigma * np.sqrt(2.0 * np.log(2.0 * l_eff)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=203,
                        help="items per class behind each AUROC (contrastive set size)")
    parser.add_argument("--out", default="results/reanalysis_scaling_law.json")
    args = parser.parse_args()

    sigma = float(np.sqrt((2 * args.n_per_class + 1) / (12 * args.n_per_class ** 2)))
    rows = []
    for path in sorted(glob.glob(str(Path(args.dir) / "*.csv"))):
        stem = re.sub(r"^.*__", "", Path(path).stem)
        if stem not in PARAMS:
            print(f"  (unmapped {stem})")
            continue
        name, params = PARAMS[stem]
        frame = pd.read_csv(path)
        col = "AUROC" if "AUROC" in frame.columns else frame.columns[-2]
        curve = frame[col].to_numpy(dtype=float)
        dist = np.abs(curve - 0.5)
        n_layers = curve.size
        # effective independent layers from lag-1 autocorrelation of the curve
        if n_layers > 3:
            r = float(np.corrcoef(curve[:-1], curve[1:])[0, 1])
        else:
            r = 0.0
        r = min(max(r, 0.0), 0.95)
        l_eff = n_layers * (1 - r) / (1 + r)
        floor = null_max(sigma, l_eff)
        observed = float(dist.max())
        rows.append({
            "model": name, "params_b": params, "n_layers": int(n_layers),
            "lag1_autocorr": r, "l_eff": float(l_eff),
            "reported_max_distance": observed,
            "null_selection_floor": floor,
            "excess_over_floor": observed - floor,
            "median_layer_distance": float(np.median(dist)),
        })

    table = pd.DataFrame(rows).sort_values("params_b").reset_index(drop=True)
    pd.set_option("display.width", 200)
    print(f"sigma(AUROC | H0, n={args.n_per_class}/class) = {sigma:.4f}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    x = np.log10(table.params_b.to_numpy())
    print(f"\n{'=' * 96}\nSCALING RELATIONSHIP: reported statistic vs corrected statistic\n{'=' * 96}")
    for label, y in (("reported  max|AUROC-0.5|", table.reported_max_distance.to_numpy()),
                     ("corrected max - null floor", table.excess_over_floor.to_numpy())):
        slope, intercept, r_v, p_v, se = stats.linregress(x, y)
        print(f"  {label:<28} slope={slope:+.4f} +/- {se:.4f}  r={r_v:+.3f}  p={p_v:.4f}")

    # is layer count a competing explanation for the reported statistic?
    layers = table.n_layers.to_numpy(dtype=float)
    print(f"\n  reported vs log10(params): r={stats.pearsonr(x, table.reported_max_distance)[0]:+.3f}")
    print(f"  reported vs n_layers      : r={stats.pearsonr(layers, table.reported_max_distance)[0]:+.3f}")
    print(f"  log10(params) vs n_layers : r={stats.pearsonr(x, layers)[0]:+.3f}  "
          f"(collinear: the two explanations are hard to separate)")
    print(f"\n  mean null floor = {table.null_selection_floor.mean():.4f}, "
          f"mean reported = {table.reported_max_distance.mean():.4f} "
          f"-> {100 * table.null_selection_floor.mean() / table.reported_max_distance.mean():.0f}% "
          f"of the average reported statistic is attributable to selection over layers")
    n_below = int((table.reported_max_distance < table.null_selection_floor).sum())
    print(f"  models whose reported statistic is BELOW its own selection floor: {n_below}/{len(table)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
