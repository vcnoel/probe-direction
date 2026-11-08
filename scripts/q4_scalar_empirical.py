"""Q4 of the controlled-transfer registration with an empirical scalar-averaging curve.

Post-submission descriptive check (docs/post-submission-checks.md), not registered; first computed
by the adversarial audit. Q4 compared the E rho^2 of averaging k direction VECTORS (empirical, on 50
random partitions of the 36 wrappers into groups of k) with the D-study value for averaging k scalar
statistics, coefficients(c, k, 1), a model-based extrapolation from the k = 1 components. This adds
the empirical scalar curve on the SAME partitions: the statistic of each wrapper per item block,
averaged within each group, then the same G-study. If the empirical scalar curve sits on the D-study
curve, the comparison Q4 made was not distorted by the D-study's assumptions.

The vector curve must reproduce Q4 in results/post_submission/controlled_transfer.json: same cells,
blocks, partition seed and draw order (controlled_transfer.ladder_summary).

    python scripts/q4_scalar_empirical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import controlled_transfer as ct  # noqa: E402
from gstudy_gtheory import coefficients, components_3way  # noqa: E402
from pod_ladders import LADDERS, N_BLOCKS, peak  # noqa: E402
from result_io import write_result  # noqa: E402

OUT = ROOT / "results" / "post_submission" / "q4_scalar_empirical.json"


def ladder(name, q4_stored):
    models = list(LADDERS[name][1])
    C = [dict(np.load(ct.CELLS / f"{name}__{m}.npz", allow_pickle=False)) for m in models]
    y, n_real = C[0]["y"], int(C[0]["n_real"])
    rng0 = np.random.default_rng(0)                               # pod_ladders blocks
    pos, neg = rng0.permutation(np.flatnonzero(y == 1)), rng0.permutation(np.flatnonzero(y == 0))
    blocks = [np.concatenate([p, q]) for p, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
    SP = [c["sadproj"].astype(np.float64) for c in C]
    xw = np.array([[[peak(y[b], sp[b][:, :, j]) for b in blocks] for j in range(n_real)] for sp in SP])
    cw = components_3way(xw)
    prng = np.random.default_rng(ct.PART_SEED)                    # Q4's partitions, in order
    curve = []
    for k in ct.KS:
        vec, sca = [], []
        for _ in range(1 if k == 1 else ct.N_PART):
            groups = (np.arange(n_real) if k == 1 else prng.permutation(n_real)).reshape(n_real // k, k)
            xg = np.array([[[peak(y[b], sp[b][:, :, g].mean(2)) for b in blocks] for g in groups]
                           for sp in SP])
            vec.append(coefficients(components_3way(xg), 1, 1)[0])
            sca.append(coefficients(components_3way(xw[:, groups, :].mean(2)), 1, 1)[0])
        curve.append({"k": k, "erho_vector_median": float(np.nanmedian(vec)),
                      "erho_scalar_empirical_median": float(np.nanmedian(sca)),
                      "erho_scalar_dstudy": float(coefficients(cw, k, 1)[0])})
    stored = {r["k"]: r["erho_vector_median"] for r in q4_stored["curve"]}
    return {"models": models, "curve": curve,
            "vector_reproduces_q4_max_abs_diff": float(max(abs(r["erho_vector_median"] - stored[r["k"]])
                                                           for r in curve)),
            "max_abs_empirical_minus_dstudy": float(max(abs(r["erho_scalar_empirical_median"] -
                                                            r["erho_scalar_dstudy"]) for r in curve))}


def main() -> None:
    ctj = json.loads((ROOT / "results" / "post_submission" / "controlled_transfer.json")
                     .read_text(encoding="utf-8"))["ladders"]
    out, pop = {}, []
    for name in LADDERS:
        out[name] = ladder(name, ctj[name]["q4"])
        pop += out[name]["models"]
        r = out[name]
        print(f"{name:16s} vector reproduces Q4 to {r['vector_reproduces_q4_max_abs_diff']:.1e}; "
              f"|empirical scalar - D-study| max {r['max_abs_empirical_minus_dstudy']:.3f}; " +
              "  ".join(f"k={c['k']}: v {c['erho_vector_median']:.3f} s {c['erho_scalar_empirical_median']:.3f} "
                        f"D {c['erho_scalar_dstudy']:.3f}" for c in r["curve"]), flush=True)
    write_result(OUT, {"ladders": out,
                       "check": "post-submission descriptive check, not registered; first computed "
                                "by the adversarial audit (docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="E rho^2 for one group (k = 1, n = 1) in model x group x item block, "
                           "median over Q4's random partitions: vector averaging, empirical scalar "
                           "averaging, and the D-study value coefficients(c, k, 1)",
                 note="the paper's statistic on its scoring set; partitions and blocks as Q4",
                 pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="ALL",
                 scoring_array="results/post_submission/controlled/*.npz sadproj",
                 data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
