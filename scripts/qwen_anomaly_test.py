"""Does the small-Qwen anomaly come from the checkpoints or from the wrapper?

Chaudhary et al. (2025) removed the Qwen family from their main analysis, reporting that smaller
Qwen checkpoints "exhibited unusually strong awareness" and citing reports of benchmark
contamination. Contamination and wrapper sensitivity predict the SAME observation here --
anomalously high scores on small checkpoints -- so the observation on its own does not discriminate
between them and cannot be cited as support for either.

It discriminates under one more measurement, which the 4 x 36 matrix already contains.

  CONTAMINATION is a property of a checkpoint. A contaminated model recognises the benchmark
  whatever prompt precedes it, so the anomaly should appear under nearly ALL wrappers.

  WRAPPER SENSITIVITY is a property of the prompt. The anomaly should appear under SOME wrappers
  and not others.

PREDICTION FIXED BEFORE RUNNING: given Kendall's W of 0.066 for the model ranking across wrappers,
the anomaly should be subset-dependent. If instead it holds under nearly every wrapper,
contamination is the better explanation of Chaudhary et al.'s observation, the paper drops its
corroboration claim, and says so.

The anomaly is operationalised two ways, both reported:
  (a) small-minus-large: mean(0.5B, 1.5B) - mean(3B, 7B) > 0
  (b) smallest-on-top:   0.5B is the argmax over the four models

    python scripts/qwen_anomaly_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    d = json.loads((ROOT / "results" / "wrapper_curves.json").read_text(encoding="utf-8"))
    M = np.array(d["matrix"])                      # [4 models, 36 wrappers]
    models, pairs = d["models"], d["pairs"]
    small, large = M[:2, :].mean(0), M[2:, :].mean(0)
    sml = small - large
    top_is_smallest = M.argmax(0) == 0
    n = M.shape[1]

    print(f"{'=' * 96}")
    print("IS THE SMALL-QWEN ANOMALY A PROPERTY OF THE CHECKPOINTS OR OF THE WRAPPER?")
    print(f"{'=' * 96}")
    print(f"  {n} wrappers, {len(models)} models, same array and statistic as the rest of the paper.")
    print("")
    print(f"  (a) small-minus-large > 0 in {int((sml > 0).sum())} of {n} wrappers "
          f"({(sml > 0).mean():.0%})")
    print(f"      difference runs {sml.min():+.3f} to {sml.max():+.3f}, median {np.median(sml):+.3f}")
    print(f"  (b) 0.5B is the top model in {int(top_is_smallest.sum())} of {n} wrappers "
          f"({top_is_smallest.mean():.0%})")
    print("")
    argmax_counts = {models[i]: int((M.argmax(0) == i).sum()) for i in range(len(models))}
    print("  which model is top, by wrapper: " +
          ", ".join(f"{k.split('-')[-1]} {v}" for k, v in argmax_counts.items()))

    frac = float((sml > 0).mean())
    uniform = frac > 0.90 or frac < 0.10
    print("")
    if uniform:
        print("  ANOMALY IS ESSENTIALLY UNIFORM ACROSS WRAPPERS. The direction of the")
        print("  small-versus-large contrast barely depends on the wrapper, so a property of the")
        print("  checkpoints -- contamination among them -- explains the observation better than")
        print("  wrapper sensitivity does. The paper drops the claim that Chaudhary et al.'s")
        print("  exclusion corroborates our instability, and reports this test instead.")
    else:
        print("  ANOMALY IS SUBSET-DEPENDENT. Whether small Qwen checkpoints out-score large ones")
        print("  depends on which wrapper is used, so the observation that prompted the exclusion is")
        print("  reproducible under some framings and absent under others. Contamination is not")
        print("  ruled out and remains a real hazard; what this shows is that the observation alone")
        print("  does not identify it, because a wrapper choice produces the same pattern.")
    print("")
    print("  Either way the honest statement is narrow: the two studies made two DIFFERENT")
    print("  observations -- scores too high on small checkpoints, and non-monotonicity with scale --")
    print("  not one observation from opposite directions.")

    out = {"n_wrappers": n, "models": models,
           "frac_small_over_large": frac,
           "n_small_over_large": int((sml > 0).sum()),
           "sml_min": float(sml.min()), "sml_max": float(sml.max()),
           "sml_median": float(np.median(sml)),
           "n_smallest_on_top": int(top_is_smallest.sum()),
           "argmax_counts": argmax_counts,
           "uniform": bool(uniform),
           "verdict": "uniform -> contamination better explains the observation" if uniform
                      else "subset-dependent -> the observation alone does not identify the cause",
           "wrappers_with_anomaly": [pairs[i] for i in np.flatnonzero(sml > 0)][:12]}
    p = ROOT / "results" / "qwen_anomaly_test.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
