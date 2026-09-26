"""Stricter G-study checks before any write-up.

Four corrections to the first pass.

1. RESTRICTED CONTRAST. The literature builds evaluation-vs-DEPLOYMENT. Of the first pass's 42
   pairs only six were eval-vs-deploy; the other 36 were eval-vs-formal-register, which no
   scaling-law paper would construct. The template-variance headline is therefore recomputed on
   the six eval x deploy pairs alone. If it survives there, the wider set is a bonus rather
   than the basis.

2. DUPLICATE STATISTIC. Folding with max(a, 1-a) makes every AUROC >= 0.5, so the absolute
   value is a no-op and peak AUROC = max|AUROC-0.5| + 0.5 exactly. They are one variable, and
   reporting both with identical variance components invites the reader to assume a bug. The
   identity is asserted and checked here, then only one is reported.

3. NULL FLOOR PER TEMPLATE. Under H0, AUROC ~ Normal(0.5, sigma) with
   sigma = sqrt((n1+n2+1)/(12 n1 n2)); maximising over L_eff effectively independent layers
   gives a positive floor even with no signal. Reporting the FRACTION of templates that clear
   their own floor separates claim types: an existence claim can survive a low generalizability
   coefficient, while ranking and localisation claims cannot.

4. DEPTH: WITHIN vs ACROSS TEMPLATE. A peak-depth range across templates includes argmax noise
   that is present even at fixed template. Bootstrapping items within each template gives the
   within-template argmax spread to compare against the across-template spread; if they are
   comparable the honest claim is that the peak layer is not identified at all, which is
   cleaner than attributing it to templates.

    python scripts/gstudy_restricted.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_analyze import ORDER, load, unit, variance_components  # noqa: E402


def auroc_curve(d, direction):
    n_layers = direction.shape[0]
    out = np.empty(n_layers)
    for L in range(n_layers):
        raw = roc_auc_score(d["y"], d["sad"][:, L, :] @ direction[L])
        out[L] = max(raw, 1.0 - raw)
    return out


def direction_for(d, f_eval, f_other):
    names = np.array(d["fname"])
    return unit(d["fr"][names == f_eval].mean(axis=0) - d["fr"][names == f_other].mean(axis=0))


def null_floor(n_per_class, curve):
    sigma = np.sqrt((2 * n_per_class + 1) / (12 * n_per_class ** 2))
    r = float(np.corrcoef(curve[:-1], curve[1:])[0, 1]) if curve.size > 3 else 0.0
    r = min(max(r, 0.0), 0.95)
    l_eff = max(curve.size * (1 - r) / (1 + r), 1.0)
    return float(sigma * np.sqrt(2.0 * np.log(2.0 * l_eff)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "results" / "gstudy_restricted.json"))
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    loaded = {m: load(m) for m in ORDER}
    loaded = {m: d for m, d in loaded.items() if d is not None}
    models = list(loaded)
    fam = next(iter(loaded.values()))["fam"]
    evals = [k for k, v in fam.items() if v == "eval"]
    deploys = [k for k, v in fam.items() if v == "deploy"]
    neutrals = [k for k, v in fam.items() if v == "neutral"]

    sets = {
        "eval x deploy (what the field builds)": list(itertools.product(evals, deploys)),
        "eval x neutral (formal register)": list(itertools.product(evals, neutrals)),
        "all non-eval (first pass)": list(itertools.product(evals, deploys + neutrals)),
    }

    report = {}
    rng = np.random.default_rng(0)
    for label, pairs in sets.items():
        stat = np.zeros((len(models), len(pairs)))
        depth = np.zeros_like(stat)
        floors = np.zeros_like(stat)
        for i, m in enumerate(models):
            d = loaded[m]
            for j, (fe, fo) in enumerate(pairs):
                curve = auroc_curve(d, direction_for(d, fe, fo))
                dist = np.abs(curve - 0.5)
                stat[i, j] = dist.max()
                depth[i, j] = (int(np.argmax(dist)) + 1) / curve.size
                floors[i, j] = null_floor(int(d["y"].sum()), curve)
        within = float(np.mean(stat.max(axis=1) - stat.min(axis=1)))
        across = float(np.mean(stat.max(axis=0) - stat.min(axis=0))) if len(models) > 1 else float("nan")
        vm, vt, ve = variance_components(stat)
        total = vm + vt + ve
        g1 = vm / total if total > 0 else float("nan")
        k80 = next((k for k in range(1, 2001) if vm > 0 and vm / (vm + (vt + ve) / k) >= 0.8), None)
        clears = float(np.mean(stat > floors))
        report[label] = {
            "n_pairs": len(pairs), "template_range": within, "model_range": across,
            "ratio": within / across if across == across and across > 0 else None,
            "var_model": vm, "var_template": vt, "var_interaction": ve,
            "share_model": vm / total, "g_one_template": g1, "k_for_g80": k80,
            "fraction_templates_above_null_floor": clears,
            "mean_null_floor": float(floors.mean()),
            "stat_min": float(stat.min()), "stat_max": float(stat.max()),
            "depth_range_across_templates": float(np.mean(depth.max(axis=1) - depth.min(axis=1))),
        }
        print(f"\n=== {label}  ({len(pairs)} pairs) ===")
        print(f"  statistic range  {stat.min():.3f} - {stat.max():.3f}   "
              f"mean null floor {floors.mean():.3f}   clears floor: {100 * clears:.0f}% of cells")
        print(f"  template range {within:.4f} | model range {across:.4f} | "
              f"ratio {within / across if across > 0 else float('nan'):.2f}x")
        print(f"  variance: model {100 * vm / total:.1f}% template {100 * vt / total:.1f}% "
              f"interaction {100 * ve / total:.1f}%")
        print(f"  E rho^2 (one template) = {g1:.3f}   k for 0.80 = {k80}")
        print(f"  peak-depth range across templates = "
              f"{report[label]['depth_range_across_templates']:.3f}")

    # ---- statistic identity check ----
    d0 = loaded[models[0]]
    c = auroc_curve(d0, direction_for(d0, evals[0], deploys[0]))
    identity = np.allclose(np.abs(c - 0.5).max() + 0.5, c.max())
    print(f"\nidentity check  peak AUROC == max|AUROC-0.5| + 0.5 : {identity} "
          f"(folded scores are >= 0.5, so the two are one variable)")

    # ---- depth: within-template item bootstrap vs across-template ----
    print("\n=== PEAK DEPTH: within-template item noise vs across-template variation ===")
    pairs = sets["eval x deploy (what the field builds)"]
    for m in models:
        d = loaded[m]
        point, within_spread = [], []
        for fe, fo in pairs:
            direction = direction_for(d, fe, fo)
            n_layers = direction.shape[0]
            proj = np.stack([d["sad"][:, L, :] @ direction[L] for L in range(n_layers)], axis=1)
            base = np.empty(n_layers)
            for L in range(n_layers):
                a = roc_auc_score(d["y"], proj[:, L])
                base[L] = abs(max(a, 1 - a) - 0.5)
            point.append((int(np.argmax(base)) + 1) / n_layers)
            ams = np.empty(args.n_boot)
            for b in range(args.n_boot):
                idx = rng.integers(0, d["y"].size, d["y"].size)
                yb = d["y"][idx]
                if yb.min() == yb.max():
                    ams[b] = np.nan
                    continue
                cb = np.empty(n_layers)
                for L in range(n_layers):
                    a = roc_auc_score(yb, proj[idx, L])
                    cb[L] = abs(max(a, 1 - a) - 0.5)
                ams[b] = (int(np.argmax(cb)) + 1) / n_layers
            lo, hi = np.nanpercentile(ams, [2.5, 97.5])
            within_spread.append(hi - lo)
        print(f"  {m:<14} across-template range {max(point) - min(point):.3f}   "
              f"mean WITHIN-template 95% argmax span {np.mean(within_spread):.3f}")
        report.setdefault("depth_detail", {})[m] = {
            "across_template_range": float(max(point) - min(point)),
            "mean_within_template_span": float(np.mean(within_spread)),
        }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
