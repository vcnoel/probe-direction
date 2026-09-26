"""On both scoring sets in this literature, content-free directions score close to the reported ones.

This is the paper's generalisation, and it arrived through a control rather than through the
experiment that was built to get it. The refusal decodability arm is void by ceiling -- the
statistic is pinned at AUROC near 1.0 because advbench-style harmful and alpaca-style harmless
instructions are near-trivially separable -- but the floor computed for that arm is informative on
its own, and it says the same thing as the evaluation-awareness floor.

Two concepts, two scoring sets, two constructions: mean-difference over answer-token activations on
multiple-choice items, against last-token differences over harmful and harmless instructions. The
margin of the reported direction over an autocorrelation-matched random direction is reported in
standard deviations of the floor, which is the only form comparable across the two.

The mechanism claim needs care. It is NOT that separable classes give a high floor: the refusal
classes are MORE separable, AUROC near 1.0 against about 0.8, and yet the standardised margin is
comparable. What matters is the ratio of the margin to the floor's spread rather than raw
separability, and the refusal margin is additionally compressed from above by the ceiling.

    python scripts/two_concept_floor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Published values recomputed by us from the released per-item scores of the depth paper.
PUBLISHED = {"qwen2.5-0.5b": 0.2115, "qwen2.5-1.5b": 0.2912,
             "qwen2.5-3b": 0.2924, "qwen2.5-7b": 0.1904}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ev = json.loads((ROOT / "results" / "null_control.json").read_text(encoding="utf-8"))
    rf = json.loads((ROOT / "results" / "refusal_decomposition_last.json")
                    .read_text(encoding="utf-8"))

    rows = []
    for m, pub in PUBLISHED.items():
        e = ev["per_model"][m]
        fl, sd = e["null_ar1_mean"], e["null_ar1_sd"]
        rows.append(("eval-awareness", m, e["concept_released"], pub, fl, sd,
                     (e["concept_released"] - fl) / sd))
    for m in rf["models"]:
        st = rf["mean_raw"][m]
        fl, sd = rf["floors"][m]["mean"], rf["floors"][m]["sd"]
        rows.append(("refusal", m, st, None, fl, sd, (st - fl) / sd))

    print(f"{'=' * 96}")
    print("CONTENT-FREE DIRECTIONS AGAINST REPORTED DIRECTIONS, TWO SCORING SETS")
    print(f"{'=' * 96}")
    print(f"  {'concept':<16}{'model':<14}{'statistic':>10}{'published':>11}{'floor':>8}"
          f"{'floor sd':>10}{'margin':>9}{'z':>7}")
    for concept, m, st, pub, fl, sd, z in rows:
        ps = f"{pub:11.3f}" if pub is not None else f"{'--':>11}"
        print(f"  {concept:<16}{m:<14}{st:10.3f}{ps}{fl:8.3f}{sd:10.3f}{st - fl:+9.3f}{z:+7.1f}")

    zs = [r[6] for r in rows]
    print("")
    print("  FLOOR PROVENANCE, because the estimate is itself implementation-dependent.")
    print("    eval-awareness: released vectors and floor both scored on")
    print("      results/gstudy/*/sad_filtered_lf.npz, rho from the released directions.")
    print("    refusal:        wrapper directions and floor both scored on")
    print("      results/refusal_v1/*/pooled.npz score_last, rho from those directions.")
    print("  Each concept's statistic and floor come from ONE array, so margins within a concept")
    print("  are internally consistent. A second eval-awareness estimate exists on the gstudy_v11")
    print("  array with rho from our wrapper directions and gives floors 0.216-0.276 against")
    print("  0.184-0.253 here. That difference of about 0.03 is comparable to the margins")
    print("  themselves, and it is this paper's thesis applied to its own null: which extraction")
    print("  and which rho you choose moves the floor by as much as the effect being measured.")
    print("  Both are reported; neither is silently mixed into a single column.")
    print(f"\n  {len(rows)} model-concept cells, standardised margin from {min(zs):+.1f} to "
          f"{max(zs):+.1f}, median {np.median(zs):+.1f}")
    print(f"  cells at or below the floor mean: {sum(z <= 0 for z in zs)}")
    print(f"  cells clearing two standard deviations: {sum(z >= 2 for z in zs)}")
    print()
    print("  On both scoring sets, directions containing no concept information score close to the")
    print("  reported directions. That is a statement about the instrument rather than about either")
    print("  subfield, and it holds across two concepts, two scoring sets and two constructions.")
    print()
    print("  The refusal arm's decodability statistic is at ceiling, AUROC near 1.0, so its margin")
    print("  is compressed from above and its decomposition is not interpretable; only the floor")
    print("  comparison is used from that arm. The mechanism is not raw separability -- the refusal")
    print("  classes are MORE separable -- but the ratio of the margin to the floor's spread.")

    out = {"pooling": "lf",
           "cells": [{"concept": c, "model": m, "statistic": st, "published": pub,
                      "floor_mean": fl, "floor_sd": sd, "z": z}
                     for c, m, st, pub, fl, sd, z in rows],
           "z_min": float(min(zs)), "z_max": float(max(zs)), "z_median": float(np.median(zs)),
           "n_cells": len(rows), "n_at_or_below_floor": int(sum(z <= 0 for z in zs)),
           "n_clearing_2sd": int(sum(z >= 2 for z in zs)),
           "floor_provenance": {
               "eval-awareness": "results/gstudy/*/sad_filtered_lf.npz; rho from released vectors",
               "refusal": "results/refusal_v1/*/pooled.npz score_last; rho from wrapper directions",
               "alternative_eval_estimate": "gstudy_v11 array, rho from wrapper directions, "
                                            "floors 0.216-0.276 vs 0.184-0.253"}}
    p = ROOT / "results" / "two_concept_floor.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
