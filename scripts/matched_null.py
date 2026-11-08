"""Matched-null audit: are the ladder findings distinguishable from content-free directions?

scripts/ladder_audit.py found that the real 36-wrapper average sits below the isotropic floor and
near the iso_rep and covariance floors, and that floor-corrected r changes sign with the null. A
real contrastive direction scoring below a random one says the nulls differ from the real
directions in geometry, not only in label content. This reads the cells of
scripts/matched_null_cells.py and

  1. characterises the real directions (norm profile, cross-layer cosine, effective number of
     layers, framing-subspace share, alignment with the scoring set's class-mean difference,
     scoring-set variance along the direction) and every null on the same properties;
  2. runs the pod_ladders pipeline on every null draw: per-wrapper r span and sign counts, eval- and
     deploy-fixed spanning counts, split-half slope reliability (Spearman-Brown over all 20 halves),
     flat and crossed G-study model shares and relative-error model shares, E rho^2 and k for
     0.80, r of the 36-wrapper average;
  3. places the real value in each null distribution (z and mid-rank percentile), and gives the
     floor-corrected r of the 36-average under each null with a bootstrap over draws of its sign;
  4. states per paper claim whether the real directions differ from the geometry-matched nulls.

The relabel null is exhaustive: all 461 non-identity splits of the 12 framings into two arms of 6,
assembled from cells of every framing pair. It is a permutation test of the evaluation/deployment
partition with the estimator and the texts held fixed; at least 18 of each split's 36 directions
are still evaluation-minus-deployment contrasts, so it is not content-free per direction.

    python scripts/matched_null.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import coefficients, components_3way  # noqa: E402
from ladder_audit import (LADDER_ORDER, N_BLOCKS, corr, crossed, erho_crossed, flat_summary,  # noqa: E402
                          grid, k_arm_080, rel_share_crossed, sb, sign_summary)
from result_io import write_result  # noqa: E402

CELLS = Path(os.environ.get("EAS_CELLS_DIR") or ROOT / "results" / "post_submission" / "cells")
OUT = Path(os.environ.get("EAS_MATCHED_OUT") or ROOT / "results" / "post_submission" / "matched_null.json")
DRAWN = ("iso", "iso_rep", "iso_xl", "cov", "matched", "matched_wide", "taskperm")
MATCHED = ("iso_xl", "matched", "matched_wide", "taskperm", "relabel")
N_FLOOR_BOOT = 2000
ALPHA = 0.05

QUANTITIES = [
    # (key, claim)
    ("r_min", "a"), ("r_max", "a"), ("r_span", "a"), ("n_positive", "a"), ("spans_both", "a"),
    ("eval_fixed_spanning", "a"), ("deploy_fixed_spanning", "a"),
    ("r_of_36_average", "b"),
    ("model_share", "c"), ("rel_model_share", "c"), ("model_share_crossed", "c"),
    ("rel_model_share_crossed", "c"), ("erho_k1", "c"), ("erho_crossed_k1", "c"),
    ("k080", "c"), ("k080_crossed_per_arm", "c"), ("mw_share", "c"),
    ("spearman_brown", "d"),
    ("mean_statistic", "level"),
]
CLAIMS = {
    "a": "the wrapper decides the sign: per-wrapper r spans both signs, within eval- and "
         "deploy-fixed rows",
    "b": "averaging over wrappers is the remedy: r of the 36-wrapper average",
    "c": "model variance is small against model x wrapper variance (G-study shares, E rho^2, k)",
    "d": "per-wrapper slopes are reliable item-wise (split-half Spearman-Brown)",
}


def pipeline(lp, full, blocks4, halves, n_e, n_d, ev_idx, dp_idx):
    """One set of 36 directions -> every quantity of QUANTITIES, and the per-model 36-average."""
    rec = sign_summary(lp, full, ev_idx, dp_idx)
    rec["r_span"] = rec["r_max"] - rec["r_min"]
    rec["spearman_brown"] = sb(lp, halves)
    rec.update(flat_summary(blocks4))
    comp = crossed(grid(blocks4, n_e, n_d))[0]
    tot = sum(comp.values())
    rec["erho_crossed_k1"] = erho_crossed(comp, 1, 1, N_BLOCKS)
    rec["rel_model_share_crossed"] = rel_share_crossed(comp)
    rec["model_share_crossed"] = comp["model"] / tot if tot > 0 else float("nan")
    kc = k_arm_080(comp)
    rec["k080_crossed_per_arm"] = kc
    rec["mean_statistic"] = float(full.mean())
    c = components_3way(blocks4)
    rec["erho_flat_36"] = coefficients(c, n_e * n_d, N_BLOCKS)[0]
    rec["erho_crossed_36"] = erho_crossed(comp, n_e, n_d, N_BLOCKS)
    return rec, full.mean(axis=1)


def as_float(v):
    if v is None:
        return np.inf                       # k for 0.80 unreachable
    return float(v)


def position(real, null):
    """Real value against a null sample: z, mid-rank percentile, summary of the null."""
    v = np.array([as_float(x) for x in null], float)
    rv = as_float(real)
    fin = v[np.isfinite(v)]
    below, ties = float((v < rv).sum()), float((v == rv).sum())
    out = {"real": None if not np.isfinite(rv) else rv,
           "null_mean": float(fin.mean()) if fin.size else None,
           "null_sd": float(fin.std(ddof=1)) if fin.size > 1 else None,
           "null_p05_p95": [float(x) for x in np.percentile(fin, [5, 95])] if fin.size else None,
           "percentile": 100.0 * (below + 0.5 * ties) / v.size,
           "n": int(v.size), "n_unreachable": int(np.isinf(v).sum())}
    sd = out["null_sd"]
    out["z"] = ((rv - out["null_mean"]) / sd if sd and np.isfinite(rv) and sd > 0 else None)
    out["outside_95"] = bool(out["percentile"] < 100 * ALPHA / 2
                             or out["percentile"] > 100 * (1 - ALPHA / 2))
    return out


def relabel_sets(arms, ev, pair_index):
    """All 6/6 splits of the 12 framings with framing 0 on the first side (mirror removed)."""
    pidx = {tuple(p): j for j, p in enumerate(pair_index.tolist())}
    n = len(arms)
    ev_set = set(range(len(ev)))
    out = []
    for rest in itertools.combinations(range(1, n), n // 2 - 1):
        A = (0,) + rest
        B = tuple(i for i in range(n) if i not in A)
        cols = [pidx[(min(a, b), max(a, b))] for a, b in itertools.product(A, B)]
        n_cross = sum((a in ev_set) != (b in ev_set) for a, b in itertools.product(A, B))
        out.append({"cols": cols, "n_cross": n_cross, "real": set(A) == ev_set})
    return out


def floor_block(lp, real_full, F, rng):
    """Real 36-average against a null: F [M, draws, 36]."""
    M = lp.size
    avg = real_full.mean(axis=1)
    fm = F.reshape(M, -1).mean(axis=1)
    fs = F.reshape(M, -1).std(axis=1, ddof=1)
    da = F.mean(axis=2)                                                   # [M, draws]
    r_fc = corr(lp, avg - fm)
    boot = []
    for _ in range(N_FLOOR_BOOT):
        b = rng.integers(0, F.shape[1], F.shape[1])
        boot.append(corr(lp, avg - F[:, b].reshape(M, -1).mean(axis=1)))
    boot = np.array(boot)
    p95 = np.percentile(F.reshape(M, -1), 95, axis=1)
    return {
        "floor_mean": fm.tolist(), "floor_r_with_log_params": corr(lp, fm),
        "real_36_average": avg.tolist(),
        "z_36_average_vs_null_36_averages": ((avg - da.mean(1)) / da.std(1, ddof=1)).tolist(),
        "percentile_36_average_vs_null_36_averages":
            (100.0 * ((da < avg[:, None]).sum(1) + 0.5 * (da == avg[:, None]).sum(1))
             / da.shape[1]).tolist(),
        "z_36_average_per_cell_sd": ((avg - fm) / fs).tolist(),
        "share_real_wrappers_above_null_p95": (real_full > p95[:, None]).mean(1).tolist(),
        "r_36_average_minus_floor": r_fc,
        "r_minus_floor_bootstrap_ci": [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])],
        "r_minus_floor_sign_stability": float(np.nanmean(np.sign(boot) == np.sign(r_fc))),
    }


def summarise_properties(props_models, models):
    """properties JSON per model -> per null: mean over models of each scalar property."""
    out = {}
    nulls = list(props_models[0])
    for k in nulls:
        rec = {}
        for key in props_models[0][k]:
            if key in ("norm_profile", "norm_rel_rms", "kbar_band_energy", "kbar_spectrum_top20"):
                continue
            per_model = [float(np.mean(p[k][key])) for p in props_models]
            rec[key] = {"mean_over_models": float(np.mean(per_model)),
                        "per_model": dict(zip(models, per_model))}
        # norm profile shape against the real one (irrelevant to AUROC, reported for completeness)
        rec["norm_profile_corr_with_real"] = float(np.mean(
            [corr(p[k]["norm_profile"], p["real"]["norm_profile"]) for p in props_models]))
        out[k] = rec
    # how far each null is from the real directions on every property (null / real, means over
    # models); no pass/fail threshold is applied here, the ratios are reported as they are
    for k in nulls:
        if k == "real":
            continue
        out[k]["ratio_to_real"] = {
            key: out[k][key]["mean_over_models"] / out["real"][key]["mean_over_models"]
            for key in ("adj_cos", "n_eff_layers", "share_top1", "share_top5", "share_top20",
                        "cos_dmu_mean", "cos_dmu_max", "scoring_var_ratio", "dprime_max")
            if out["real"][key]["mean_over_models"] != 0}
        out[k]["cos_real_over_subspace_floor"] = (
            out[k]["cos_real"]["mean_over_models"]
            / max(out[k]["cos_between_draws"]["mean_over_models"],
                  out["real"]["cos_between_wrappers"]["mean_over_models"]))
    out["real"]["norm_rel_rms_per_layer"] = {m: p["real"]["norm_rel_rms"]
                                             for m, p in zip(models, props_models)}
    out["real"]["kbar_band_energy_dyadic"] = {m: p["real"]["kbar_band_energy"]
                                              for m, p in zip(models, props_models)}
    out["real"]["kbar_spectrum_top20"] = {m: p["real"]["kbar_spectrum_top20"]
                                          for m, p in zip(models, props_models)}
    return out


def run(name, rng):
    z = dict(np.load(CELLS / f"matched_{name}.npz", allow_pickle=False))
    old = dict(np.load(CELLS / f"{name}.npz", allow_pickle=False))
    lp = np.log10(z["params_b"].astype(float))
    models = [str(m) for m in z["models"]]
    arms = [str(a) for a in z["arms"]]
    ev = [a for a in arms if a.startswith("eval_")]
    dp = [a for a in arms if a.startswith("deploy_")]
    n_e, n_d = len(ev), len(dp)
    ev_idx = [list(range(i * n_d, (i + 1) * n_d)) for i in range(n_e)]
    dp_idx = [list(range(j, n_e * n_d, n_d)) for j in range(n_d)]
    cfg = json.loads(str(z["config"]))

    rf = z["real_full"].astype(float)
    repro = {"max_abs_diff_full_vs_ladder_audit_cells": float(np.abs(rf - old["real_full"]).max()),
             "max_abs_diff_blocks4": float(np.abs(z["real_blocks4"] - old["real_blocks4"]).max()),
             "max_abs_diff_halves": float(np.abs(z["real_halves"] - old["real_halves"]).max())}
    assert repro["max_abs_diff_full_vs_ladder_audit_cells"] < 2e-3, repro
    real, _ = pipeline(lp, rf, z["real_blocks4"].astype(float),
                       z["real_halves"].astype(float).transpose(1, 2, 0, 3), n_e, n_d,
                       ev_idx, dp_idx)

    draws, fulls = {}, {}
    for k in DRAWN:
        F = z[f"{k}_full"].astype(float)                    # [M, draws, 36]
        Bk = z[f"{k}_blocks4"].astype(float)                # [M, draws, 36, 4]
        H = z[f"{k}_halves"].astype(float)                  # [M, splits, 2, draws, 36]
        recs = [pipeline(lp, F[:, b], Bk[:, b], H[:, :, :, b].transpose(1, 2, 0, 3), n_e, n_d,
                         ev_idx, dp_idx)[0] for b in range(F.shape[1])]
        draws[k], fulls[k] = recs, F
    # relabel: exhaustive splits of the framings into two arms of 6
    sets = relabel_sets(arms, ev, z["pair_index"])
    PF, PB, PH = (z["pairs_full"].astype(float), z["pairs_blocks4"].astype(float),
                  z["pairs_halves"].astype(float))
    rel, rel_full, n_cross = [], [], []
    for s in sets:
        if s["real"]:
            # the identity split must reproduce the real cells (up to sign, which folding removes)
            assert np.abs(PF[:, s["cols"]] - rf).max() < 2e-3
            continue
        c = s["cols"]
        rec, _ = pipeline(lp, PF[:, c], PB[:, c], PH[:, :, :, c].transpose(1, 2, 0, 3), n_e, n_d,
                          ev_idx, dp_idx)
        rel.append(rec)
        rel_full.append(PF[:, c])
        n_cross.append(s["n_cross"])
    draws["relabel"] = rel
    fulls["relabel"] = np.stack(rel_full, axis=1)
    n_cross = np.array(n_cross)
    bal = n_cross == n_cross.min()
    draws["relabel_balanced"] = [r for r, b in zip(rel, bal) if b]
    fulls["relabel_balanced"] = fulls["relabel"][:, bal]

    keys = [q for q, _ in QUANTITIES] + ["erho_flat_36", "erho_crossed_36"]
    comparison = {}
    for k, recs in draws.items():
        comparison[k] = {q: position(real[q], [r[q] for r in recs]) for q in keys}
        comparison[k]["n_draws"] = len(recs)
    floors = {k: floor_block(lp, rf, F, rng) for k, F in fulls.items()}
    props = summarise_properties([json.loads(str(p)) for p in z["properties"]], models)

    verdict = {}
    for claim in CLAIMS:
        qs = [q for q, c in QUANTITIES if c == claim]
        verdict[claim] = {
            k: {q: {"percentile": comparison[k][q]["percentile"],
                    "outside_95": comparison[k][q]["outside_95"]} for q in qs}
            for k in draws}
    verdict["b_floor_corrected_r"] = {k: {"r": floors[k]["r_36_average_minus_floor"],
                                          "sign_stability": floors[k]["r_minus_floor_sign_stability"]}
                                      for k in floors}
    signs = {k: np.sign(v["r"]) for k, v in verdict["b_floor_corrected_r"].items()}
    verdict["b_floor_corrected_r_sign_same_under_all_nulls"] = bool(len(set(signs.values())) == 1)
    verdict["b_floor_corrected_r_sign_same_under_matched_nulls"] = bool(
        len({signs[k] for k in MATCHED}) == 1)

    return {"models": models, "real": real, "comparison": comparison, "floors": floors,
            "properties": props, "verdict": verdict, "reproduces_cached_real_cells": repro,
            "relabel_n_cross_distribution": {int(v): int((n_cross == v).sum())
                                             for v in np.unique(n_cross)},
            "config": cfg}


def cross_ladder(out):
    """Per null and quantity: in how many ladders the real value is below 2.5% / above 97.5%."""
    tab = {}
    nulls = list(next(iter(out.values()))["comparison"])
    for k in nulls:
        tab[k] = {}
        for q, _ in QUANTITIES:
            pcts = {n: out[n]["comparison"][k][q]["percentile"] for n in out}
            tab[k][q] = {"percentile_per_ladder": pcts,
                         "n_ladders_below_2.5": int(sum(p < 2.5 for p in pcts.values())),
                         "n_ladders_above_97.5": int(sum(p > 97.5 for p in pcts.values()))}
    return tab


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")
    ladders = [n for n in LADDER_ORDER if (CELLS / f"matched_{n}.npz").exists()]
    if not ladders:
        raise SystemExit("no matched cells; run scripts/matched_null_cells.py")
    out, pop = {}, []
    for name in ladders:
        out[name] = run(name, np.random.default_rng(0))
        pop += out[name]["models"]
        c, r = out[name]["comparison"], out[name]["real"]
        print(f"  {name}: r36 {r['r_of_36_average']:+.3f}  SB {r['spearman_brown']:.3f}  "
              f"model share {r['model_share']:.3f}")
        for k in c:
            print(f"     {k:16s} pct r36 {c[k]['r_of_36_average']['percentile']:5.1f}  "
                  f"SB {c[k]['spearman_brown']['percentile']:5.1f}  "
                  f"mshare {c[k]['model_share']['percentile']:5.1f}  "
                  f"span {c[k]['r_span']['percentile']:5.1f}  "
                  f"level {c[k]['mean_statistic']['percentile']:5.1f}  "
                  f"r-floor {out[name]['floors'][k]['r_36_average_minus_floor']:+.2f} "
                  f"({out[name]['floors'][k]['r_minus_floor_sign_stability']:.2f})", flush=True)
    conv = dict(pooling="validated for scoring and for the direction arrays "
                        "(Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                layers="ALL", scoring_array="results/post_submission/cells/matched_*.npz, from "
                "results/pod_*/*.npz and results/gstudy via scripts/matched_null_cells.py",
                data_root="pod", model_population=sorted(set(pop)))
    write_result(OUT, {"ladders": out, "cross_ladder": cross_ladder(out), "claims": CLAIMS,
                       "matched_nulls": list(MATCHED)},
                 kind="measured",
                 statistic="the pod_ladders pipeline on geometry-matched content-free directions; "
                           "real value placed in each null by z and mid-rank percentile",
                 note="post-submission audit; not part of the submitted manuscript or its "
                      "recorded predictions", **conv)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
