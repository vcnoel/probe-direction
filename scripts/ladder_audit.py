"""Post-submission audit of the crossed wrapper design on every ladder, from the cached cells.

Reads results/post_submission/cells/<ladder>.npz (scripts/ladder_audit_cells.py) and answers the
audit's questions with the same statistic, blocks and halves as scripts/pod_ladders.py. Every
quantity below is computed on every ladder whose cells exist; nothing is selected per ladder.

  averaged_estimator.json   The recommended remedy, the 36-wrapper average, as an estimator of the
                            scaling correlation: r with log10 parameters, with intervals from
                            resampling items, from resampling arms (6 evaluation and 6 deployment
                            framings with replacement), and from both; an exact permutation p over
                            the models; leave-one-arm-out and leave-one-model-out values; the same r
                            for null directions and after subtracting each model's null floor. Also
                            how the per-wrapper slope splits into evaluation-arm and deployment-arm
                            main effects and their interaction, against item-sampling noise.
  crossed_gstudy.json       The four-facet crossed decomposition model x eval arm x deploy arm x
                            item block with every interaction, against the flat three-facet one;
                            E rho^2 and k for 0.80 under both; the model share of RELATIVE error
                            variance (model against model x facet terms only); joint bootstrap
                            intervals; and the dependence on block size of the maximum over layers
                            (Eq. 1 treats the full-set score as the mean of block scores).
  null_directions.json      The full pipeline (sign span, split-half slope reliability, G-study
                            shares, E rho^2, k for 0.80) on five kinds of direction carrying no
                            label information, against the real directions; and each model's
                            content-free floor with the standardised excess of the real wrappers.
                            The paired sign flip is reported but is NOT content-free here (|cos|
                            with the real direction ~0.99); iso and iso_rep are, and bracket the
                            floor; cov is label-free but lives in the framing subspace.
  arm_disjoint.json         Sign and ranking agreement between averages over arm-disjoint wrapper
                            sets (3 eval x 3 deploy against the other 3 x 3, all 200 splits),
                            beside random disjoint halves, and a simulation of why random disjoint
                            halves of a fixed pool whose mean slope is near zero agree on the sign
                            LESS often than chance as k grows.
  prediction_scoring.json   The pod/PREDICTIONS.md section 4 predictions scored on BOTH Qwen3.5
                            renderings (and on Qwen3 for reference), with bootstrap probabilities.

    python scripts/ladder_audit.py
"""

from __future__ import annotations

import itertools
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import coefficients, components_3way, total  # noqa: E402
from implementation_facet import components_nway  # noqa: E402
from result_io import write_result  # noqa: E402

CELLS = ROOT / "results" / "post_submission" / "cells"
OUTDIR = ROOT / "results" / "post_submission"
LADDER_ORDER = ["qwen2.5_paper", "qwen3_nothink", "qwen3_think", "qwen3.5_nothink",
                "qwen3.5_think", "olmo2"]
N_ARM_BOOT = 4000
N_BLOCKS = 4
TARGET = 0.80
FACETS = ["model", "eval", "deploy", "item"]


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(y) == 0 or np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_cols(lp, X):
    """Pearson r of lp with every column of X [models, cols]."""
    a = lp - lp.mean()
    B = X - X.mean(axis=0)
    den = np.sqrt((a ** 2).sum() * (B ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a @ B) / den


def slopes(lp, X):
    a = lp - lp.mean()
    return (a @ (X - X.mean(axis=0))) / (a ** 2).sum()


def q(v, p=(2.5, 97.5)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, p)] if v.size else [None, None]


def spans_both(r, idx):
    return bool(max(r[i] for i in idx) > 0 > min(r[i] for i in idx))


def grid(x, n_e, n_d):
    """[M, 36, ...] -> [M, n_e, n_d, ...]; pairs are itertools.product(eval, deploy)."""
    return x.reshape(x.shape[0], n_e, n_d, *x.shape[2:])


def crossed(x4):
    comp, trunc, _ = components_nway(x4, FACETS)
    return comp, trunc


def erho_crossed(comp, kE, kD, n):
    sm, den = comp["model"], 0.0
    for key, v in comp.items():
        parts = key.split("x")
        if "model" not in parts or key == "model":
            continue
        div = (kE if "eval" in parts else 1) * (kD if "deploy" in parts else 1) * \
              (n if "item" in parts else 1)
        den += v / div
    return sm / (sm + den) if sm + den > 0 else float("nan")


def k_arm_080(comp, n=N_BLOCKS, kmax=5000):
    """Smallest k framings PER ARM (k x k wrappers) with crossed E rho^2 >= 0.80, or None."""
    if not erho_crossed(comp, kmax, kmax, n) >= TARGET:
        return None
    lo, hi = 1, kmax
    while lo < hi:
        mid = (lo + hi) // 2
        if erho_crossed(comp, mid, mid, n) >= TARGET:
            hi = mid
        else:
            lo = mid + 1
    return lo


def k_flat_080(c, n=N_BLOCKS, kmax=100000):
    if not coefficients(c, kmax, n)[0] >= TARGET:
        return None
    lo, hi = 1, kmax
    while lo < hi:
        mid = (lo + hi) // 2
        if coefficients(c, mid, n)[0] >= TARGET:
            hi = mid
        else:
            lo = mid + 1
    return lo


def rel_share_flat(c):
    den = c["s_m"] + c["s_mt"] + c["s_mh"] + c["s_mth"]
    return c["s_m"] / den if den > 0 else float("nan")


def rel_share_crossed(comp):
    den = sum(v for k, v in comp.items() if "model" in k.split("x"))
    return comp["model"] / den if den > 0 else float("nan")


def flat_summary(xb):
    """[M, 36, 4] -> dict of the flat G-study quantities the paper prints."""
    c = components_3way(xb)
    tot = total(c)
    return {"model_share": c["s_m"] / tot, "mw_share": c["s_mt"] / tot, "mi_share": c["s_mh"] / tot,
            "rel_model_share": rel_share_flat(c),
            "erho_k1": coefficients(c, 1, N_BLOCKS)[0],
            "k080": k_flat_080(c),
            "mw_over_mi": c["s_mt"] / c["s_mh"] if c["s_mh"] > 0 else float("inf"),
            "model_below_mw": bool(c["s_m"] < c["s_mt"])}


def sb(lp, halves):
    """halves [n_splits, 2, M, 36] -> Spearman-Brown of split-half slope correlation."""
    sc = [corr(slopes(lp, halves[s, 0]), slopes(lp, halves[s, 1])) for s in range(halves.shape[0])]
    m = float(np.nanmean(sc))
    return 2 * m / (1 + m)


def sign_summary(lp, full, ev_idx, dp_idx):
    r = corr_cols(lp, full)
    return {"r_min": float(np.nanmin(r)), "r_max": float(np.nanmax(r)),
            "n_positive": int((r > 0).sum()), "n_negative": int((r < 0).sum()),
            "eval_fixed_spanning": int(sum(spans_both(r, i) for i in ev_idx)),
            "deploy_fixed_spanning": int(sum(spans_both(r, i) for i in dp_idx)),
            "spans_both": bool(np.nanmax(r) > 0 > np.nanmin(r)),
            "r_of_36_average": corr(lp, full.mean(axis=1))}


# ------------------------------------------------------------------------------------------------
# per-ladder analyses
# ------------------------------------------------------------------------------------------------

def averaged_estimator(z, lp, rng):
    full = z["real_full"].astype(float)                     # [M, 36]
    M = full.shape[0]
    n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
    g = grid(full, n_e, n_d)
    avg = full.mean(axis=1)
    r36 = corr(lp, avg)

    # items: the bootstrap replicates of ladder_audit_cells (same resampled items for every model)
    boot = z["boot_full"].astype(float)                     # [M, B, 36]
    r_items = np.array([corr(lp, boot[:, b].mean(axis=1)) for b in range(boot.shape[1])])

    # arms: resample 6 evaluation and 6 deployment framings with replacement
    def arm_draw(G):
        e = rng.integers(0, n_e, n_e)
        d = rng.integers(0, n_d, n_d)
        return G[:, e][:, :, d].reshape(M, -1).mean(axis=1)
    r_arms = np.array([corr(lp, arm_draw(g)) for _ in range(N_ARM_BOOT)])
    # both: one arm draw per item replicate, several per replicate
    r_both = []
    for b in range(boot.shape[1]):
        Gb = grid(boot[:, b], n_e, n_d)
        for _ in range(20):
            r_both.append(corr(lp, arm_draw(Gb)))
    r_both = np.array(r_both)

    # exact permutation over the models (which model carries which parameter count)
    perms = list(itertools.permutations(range(M)))
    rp = np.array([corr(lp[list(p)], avg) for p in perms])
    p_one = float((rp >= r36 - 1e-12).mean())
    p_two = float((np.abs(rp) >= abs(r36) - 1e-12).mean())

    # leave one arm out, leave one model out
    loao = {}
    for i, a in enumerate(z["eval_arms"]):
        loao[f"drop {a}"] = corr(lp, np.delete(g, i, axis=1).reshape(M, -1).mean(axis=1))
    for i, a in enumerate(z["deploy_arms"]):
        loao[f"drop {a}"] = corr(lp, np.delete(g, i, axis=2).reshape(M, -1).mean(axis=1))
    lomo = {str(z["models"][i]): corr(np.delete(lp, i), np.delete(avg, i)) for i in range(M)}

    # arm main effects vs interaction, on the per-wrapper slope (and on r for the sign reading)
    sl = slopes(lp, full).reshape(n_e, n_d)
    r = corr_cols(lp, full).reshape(n_e, n_d)
    gm = sl.mean()
    ae, ad = sl.mean(axis=1) - gm, sl.mean(axis=0) - gm
    inter = sl - gm - ae[:, None] - ad[None, :]
    ss_e, ss_d, ss_i = n_d * (ae ** 2).sum(), n_e * (ad ** 2).sum(), (inter ** 2).sum()
    ss_t = ss_e + ss_d + ss_i
    # item-sampling noise of each wrapper's full-set slope, from the item bootstrap
    boot_sl = np.stack([slopes(lp, boot[:, b]) for b in range(boot.shape[1])])   # [B, 36]
    noise_var = float(boot_sl.var(axis=0, ddof=1).mean())
    ms_i = ss_i / ((n_e - 1) * (n_d - 1))
    ms_e, ms_d = ss_e / (n_e - 1), ss_d / (n_d - 1)
    # arm-level permutation test of the evaluation main effect: shuffle cells within deploy columns
    perm_e, perm_d = [], []
    for _ in range(4000):
        s2 = np.stack([rng.permutation(sl[:, j]) for j in range(n_d)], axis=1)
        perm_e.append(n_d * ((s2.mean(axis=1) - s2.mean()) ** 2).sum())
        s3 = np.stack([rng.permutation(sl[i, :]) for i in range(n_e)], axis=0)
        perm_d.append(n_e * ((s3.mean(axis=0) - s3.mean()) ** 2).sum())
    arms = {
        "slope_ss_share": {"eval_main": ss_e / ss_t, "deploy_main": ss_d / ss_t,
                           "interaction": ss_i / ss_t},
        "mean_square": {"eval_main": ms_e, "deploy_main": ms_d, "interaction": ms_i,
                        "item_noise_per_wrapper": noise_var},
        "interaction_over_item_noise": ms_i / noise_var if noise_var > 0 else None,
        "eval_main_perm_p": float((np.array(perm_e) >= ss_e - 1e-15).mean()),
        "deploy_main_perm_p": float((np.array(perm_d) >= ss_d - 1e-15).mean()),
        "per_eval_arm": {str(a): {"mean_r": float(r[i].mean()), "n_positive": int((r[i] > 0).sum()),
                                  "r": [float(v) for v in r[i]]}
                         for i, a in enumerate(z["eval_arms"])},
        "per_deploy_arm": {str(a): {"mean_r": float(r[:, j].mean()),
                                    "n_positive": int((r[:, j] > 0).sum())}
                           for j, a in enumerate(z["deploy_arms"])},
    }

    # bootstrap-robust sign span: a wrapper counts as positive (negative) only if its r keeps that
    # sign in at least 95% of item-bootstrap replicates
    rb = np.stack([corr_cols(lp, boot[:, b]) for b in range(boot.shape[1])])     # [B, 36]
    p_pos = (rb > 0).mean(axis=0)
    firm = np.where(p_pos >= 0.95, 1, np.where(p_pos <= 0.05, -1, 0)).reshape(n_e, n_d)
    robust = {
        "n_firm_positive": int((firm > 0).sum()), "n_firm_negative": int((firm < 0).sum()),
        "eval_fixed_spanning_firm": int(sum((row > 0).any() and (row < 0).any() for row in firm)),
        "deploy_fixed_spanning_firm": int(sum((col > 0).any() and (col < 0).any()
                                              for col in firm.T)),
        "p_positive_per_wrapper": {f"{a}/{b}": float(p) for (a, b), p in
                                   zip(z["pairs"], p_pos)},
    }
    return {
        "models": [str(m) for m in z["models"]], "r_36_average": r36,
        "ci_items": q(r_items), "ci_arms": q(r_arms), "ci_items_and_arms": q(r_both),
        "p_positive_items": float((r_items > 0).mean()),
        "p_positive_arms": float((r_arms > 0).mean()),
        "p_positive_items_and_arms": float((r_both > 0).mean()),
        "n_boot_items": int(boot.shape[1]), "n_boot_arms": N_ARM_BOOT,
        "n_boot_items_and_arms": int(r_both.size),
        "permutation_p_one_sided": p_one, "permutation_p_two_sided": p_two,
        "n_permutations": len(perms),
        "leave_one_arm_out": loao,
        "leave_one_arm_out_range": [min(loao.values()), max(loao.values())],
        "leave_one_arm_out_all_same_sign": bool(all(np.sign(v) == np.sign(r36)
                                                    for v in loao.values())),
        "leave_one_model_out": lomo,
        "arm_structure": arms, "bootstrap_robust_sign_span": robust,
    }


def crossed_gstudy(z, rng):
    n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
    x = z["real_blocks4"].astype(float)                     # [M, 36, 4]
    comp, trunc = crossed(grid(x, n_e, n_d))
    tot = sum(comp.values())
    c = components_3way(x)
    ctot = total(c)
    out = {
        "components_crossed": comp, "truncated_crossed": trunc,
        "shares_crossed": {k: v / tot for k, v in comp.items()},
        "components_flat": {k: v for k, v in c.items() if not k.startswith("_")},
        "shares_flat": {k: v / ctot for k, v in c.items() if not k.startswith("_")},
        "erho_one_wrapper_full_items": {"flat": coefficients(c, 1, N_BLOCKS)[0],
                                        "crossed": erho_crossed(comp, 1, 1, N_BLOCKS)},
        "erho_36_wrappers_full_items": {"flat": coefficients(c, n_e * n_d, N_BLOCKS)[0],
                                        "crossed": erho_crossed(comp, n_e, n_d, N_BLOCKS)},
        "k080": {"flat_wrappers": k_flat_080(c),
                 "crossed_framings_per_arm": k_arm_080(comp)},
        "erho_asymptote_many_wrappers": {"flat": coefficients(c, 10 ** 7, N_BLOCKS)[0],
                                         "crossed": erho_crossed(comp, 10 ** 5, 10 ** 5, N_BLOCKS)},
        "relative_model_share": {"flat": rel_share_flat(c), "crossed": rel_share_crossed(comp)},
        "total_share_model": {"flat": c["s_m"] / ctot, "crossed": comp["model"] / tot},
        "model_vs_mw": {"s_m": c["s_m"], "s_mt": c["s_mt"], "s_m_exceeds_s_mt": bool(c["s_m"] > c["s_mt"])},
    }
    kc = out["k080"]["crossed_framings_per_arm"]
    out["k080"]["crossed_wrappers"] = kc * kc if kc else None
    out["k080"]["crossed_texts_to_write"] = 2 * kc if kc else None

    # joint bootstrap: item replicates of ladder_audit_cells x one arm draw each
    bb = z["boot_blocks4"].astype(float)                    # [M, B, 36, 4]
    rec = {k: [] for k in ("erho1_flat", "erho1_crossed", "erho36_crossed", "model_share_flat",
                           "rel_share_flat", "rel_share_crossed", "mw_share_flat",
                           "model_below_mw", "mw_over_mi")}
    for b in range(bb.shape[1]):
        e = rng.integers(0, n_e, n_e)
        d = rng.integers(0, n_d, n_d)
        g4 = grid(bb[:, b], n_e, n_d)[:, e][:, :, d]
        cb, _ = crossed(g4)
        fb = components_3way(g4.reshape(g4.shape[0], -1, N_BLOCKS))
        ft = total(fb)
        rec["erho1_flat"].append(coefficients(fb, 1, N_BLOCKS)[0])
        rec["erho1_crossed"].append(erho_crossed(cb, 1, 1, N_BLOCKS))
        rec["erho36_crossed"].append(erho_crossed(cb, n_e, n_d, N_BLOCKS))
        rec["model_share_flat"].append(fb["s_m"] / ft)
        rec["mw_share_flat"].append(fb["s_mt"] / ft)
        rec["rel_share_flat"].append(rel_share_flat(fb))
        rec["rel_share_crossed"].append(rel_share_crossed(cb))
        rec["model_below_mw"].append(float(fb["s_m"] < fb["s_mt"]))
        rec["mw_over_mi"].append(fb["s_mt"] / fb["s_mh"] if fb["s_mh"] > 0 else np.inf)
    out["bootstrap"] = {k: {"ci": q(v), "mean": float(np.mean(np.array(v)[np.isfinite(v)]))}
                        for k, v in rec.items()}
    out["bootstrap"]["p_model_share_below_mw"] = float(np.mean(rec["model_below_mw"]))
    out["bootstrap"]["p_mw_over_mi_at_least_2"] = float(np.mean(np.array(rec["mw_over_mi"]) >= 2))
    out["bootstrap"]["n"] = int(bb.shape[1])

    # max-over-layers selection bias as a function of block size (Eq. 1 treats the full-set score
    # as the mean of block scores; with a maximum over layers it is not)
    full = z["real_full"].astype(float)
    bs = {}
    for nb, key in ((2, "real_blocks2"), (4, "real_blocks4"), (8, "real_blocks8")):
        xb = z[key].astype(float)
        cb = components_3way(xb)
        bs[str(nb)] = {
            "items_per_block": int(round(len(z["y"]) / nb)),
            "mean_block_statistic": float(xb.mean()),
            "mean_full_statistic": float(full.mean()),
            "bias_mean_of_blocks_minus_full": float((xb.mean(axis=2) - full).mean()),
            "bias_relative": float((xb.mean(axis=2) - full).mean() / full.mean()),
            "corr_mean_of_blocks_with_full_across_cells": corr(full.ravel(), xb.mean(axis=2).ravel()),
            "model_share": cb["s_m"] / total(cb),
            "erho_one_wrapper_full_items_via_eq1": coefficients(cb, 1, nb)[0],
            "k080_via_eq1": k_flat_080(cb, n=nb),
        }
    out["block_size"] = bs
    return out


def null_pipeline(z, lp, ev_idx, dp_idx):
    n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
    M = lp.size
    real_full = z["real_full"].astype(float)
    halves = z["real_halves"].astype(float).transpose(1, 2, 0, 3)      # [splits, 2, M, 36]
    res = {"real": {**sign_summary(lp, real_full, ev_idx, dp_idx),
                    "spearman_brown": sb(lp, halves),
                    # the nulls use the first 10 of the same 20 splits
                    "spearman_brown_first10": sb(lp, halves[:10]),
                    **flat_summary(z["real_blocks4"].astype(float)),
                    "erho_crossed_k1": erho_crossed(crossed(grid(z["real_blocks4"].astype(float),
                                                                  n_e, n_d))[0], 1, 1, N_BLOCKS)}}
    floors = {}
    for k in ("signflip", "permute", "cov", "iso", "iso_rep"):
        if f"{k}_full" not in z:
            continue
        F = z[f"{k}_full"].astype(float)                    # [M, draws, 36]
        Bk = z[f"{k}_blocks4"].astype(float)                # [M, draws, 36, 4]
        H = z[f"{k}_halves"].astype(float)                  # [M, splits, 2, draws, 36]
        draws = []
        for b in range(F.shape[1]):
            full = F[:, b]
            rec = sign_summary(lp, full, ev_idx, dp_idx)
            rec["spearman_brown"] = sb(lp, H[:, :, :, b].transpose(1, 2, 0, 3))
            rec.update(flat_summary(Bk[:, b]))
            rec["erho_crossed_k1"] = erho_crossed(crossed(grid(Bk[:, b], n_e, n_d))[0], 1, 1,
                                                  N_BLOCKS)
            draws.append(rec)
        summ = {}
        for key in draws[0]:
            v = np.array([d[key] if d[key] is not None else np.nan for d in draws], float)
            fin = v[np.isfinite(v)]
            summ[key] = {"mean": float(fin.mean()) if fin.size else None,
                         "sd": float(fin.std(ddof=1)) if fin.size > 1 else None,
                         "p05_p95": q(fin, (5, 95)) if fin.size else None,
                         "n_finite": int(fin.size)}
            rv = res["real"].get(key)
            if rv is not None and fin.size and not isinstance(rv, bool):
                summ[key]["share_of_draws_at_or_above_real"] = float((fin >= rv).mean())
        summ["n_draws"] = len(draws)
        if f"cos_{k}" in z:
            # per model: median |cos| between a null draw and the real wrapper direction it
            # replaces. Near 1 means the "null" still is the real direction (signflip).
            summ["median_abs_cos_with_real_direction"] = [float(v) for v in z[f"cos_{k}"]]
        res[k] = summ
        # content-free floor per model: the null statistic pooled over draws and wrappers
        fm, fs = F.reshape(M, -1).mean(axis=1), F.reshape(M, -1).std(axis=1, ddof=1)
        zr = (real_full - fm[:, None]) / fs[:, None]
        avg = real_full.mean(axis=1)
        floors[k] = {
            "floor_mean": fm.tolist(), "floor_sd": fs.tolist(),
            "floor_r_with_log_params": corr(lp, fm),
            "real_36_average": avg.tolist(),
            "standardised_excess_of_36_average": ((avg - fm) / fs).tolist(),
            "standardised_excess_per_wrapper_median": np.median(zr, axis=1).tolist(),
            "share_of_wrappers_clearing_2sd": (zr > 2).mean(axis=1).tolist(),
            "r_36_average_minus_floor": corr(lp, avg - fm),
            "r_36_average_raw": corr(lp, avg),
        }
    return res, floors


def arm_disjoint(z, lp, rng):
    full = z["real_full"].astype(float)
    M = full.shape[0]
    n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
    g = grid(full, n_e, n_d)
    ref = full.mean(axis=1)
    r_ref = corr(lp, ref)
    seen, agree, agree_ref, taus = set(), [], [], []
    for E1 in combinations(range(n_e), n_e // 2):
        E2 = tuple(i for i in range(n_e) if i not in E1)
        for D1 in combinations(range(n_d), n_d // 2):
            D2 = tuple(i for i in range(n_d) if i not in D1)
            key = frozenset([(E1, D1), (E2, D2)])
            if key in seen:
                continue
            seen.add(key)
            a = g[:, list(E1)][:, :, list(D1)].reshape(M, -1).mean(axis=1)
            b = g[:, list(E2)][:, :, list(D2)].reshape(M, -1).mean(axis=1)
            ra, rb = corr(lp, a), corr(lp, b)
            agree.append(np.sign(ra) == np.sign(rb))
            agree_ref.append((np.sign(ra) == np.sign(r_ref)) and (np.sign(rb) == np.sign(r_ref)))
            taus.append(kendalltau(a, b).statistic)
    # random disjoint wrapper sets of the same size (9) and of half the pool (18), as
    # wrapper_subsets.py draws them
    rnd = {}
    nw = full.shape[1]
    for k in (9, 18):
        ag, ts = [], []
        for _ in range(4000):
            p = rng.permutation(nw)
            a, b = full[:, p[:k]].mean(axis=1), full[:, p[k:2 * k]].mean(axis=1)
            ag.append(np.sign(corr(lp, a)) == np.sign(corr(lp, b)))
            ts.append(kendalltau(a, b).statistic)
        rnd[str(k)] = {"sign_agreement": float(np.mean(ag)), "mean_tau": float(np.mean(ts))}
    sl = slopes(lp, full)
    return {"n_splits": len(seen), "r_36_average": r_ref,
            "pool_mean_slope_z": float(abs(sl.mean()) / (sl.std(ddof=1) / np.sqrt(sl.size))),
            "arm_disjoint_3x3_sign_agreement": float(np.mean(agree)),
            "arm_disjoint_3x3_both_match_36_average": float(np.mean(agree_ref)),
            "arm_disjoint_3x3_mean_tau": float(np.mean(taus)),
            "random_disjoint": rnd}


def finite_pool_simulation(rng, n_models=4, n_wrappers=36, n_pools=600, n_draws=60):
    """Why disjoint halves of ONE fixed pool can agree on the sign less than half the time.

    Each wrapper's statistic vector is a wrapper-specific deviation plus a slope component whose
    population mean is beta (in units of the across-wrapper slope sd). For a fixed pool of 36
    wrappers, two disjoint k-subsets are drawn as wrapper_subsets.py draws them. At k = 18 the two
    halves partition the pool, so their averages sum to twice the pool average; when the POOL's
    own mean slope is near zero the halves are pushed to opposite signs, and agreement falls BELOW
    one half as k grows. Pools are binned by their own standardised mean slope
    z = mean / (sd / sqrt(36)); Qwen2.5 sits in the lowest bin. Independent k-samples from the
    population (no shared pool) give the replication rate a reader would actually want."""
    lp = np.linspace(-0.3, 0.9, n_models)
    a = (lp - lp.mean()) / np.sqrt(((lp - lp.mean()) ** 2).sum())
    ks = [1, 2, 4, 6, 9, 12, 15, 18]
    bins = {"|z|<0.5": (0.0, 0.5), "0.5<=|z|<2": (0.5, 2.0), "|z|>=2": (2.0, np.inf)}
    out = {"k": ks}
    for beta in (0.0, 0.3):
        def draw(n):
            w = rng.standard_normal((n_models, n)) * 0.5
            w = w - np.outer(a, a @ w)
            return w + np.outer(a, beta + rng.standard_normal(n))
        rows, zs, indep = [], [], {k: [] for k in ks}
        for _ in range(n_pools):
            pool = draw(n_wrappers)
            sl = a @ pool
            zs.append(abs(sl.mean()) / (sl.std(ddof=1) / np.sqrt(n_wrappers)))
            curve = []
            for k in ks:
                hits = 0
                for _ in range(n_draws):
                    p = rng.permutation(n_wrappers)
                    hits += np.sign(sl[p[:k]].mean()) == np.sign(sl[p[k:2 * k]].mean())
                curve.append(hits / n_draws)
            rows.append(curve)
            for k in ks:
                indep[k].append(float(np.sign((a @ draw(k)).mean()) == np.sign((a @ draw(k)).mean())))
        rows, zs = np.array(rows), np.array(zs)
        rec = {"independent_samples_sign_agreement": [float(np.mean(indep[k])) for k in ks]}
        for bname, (lo, hi) in bins.items():
            sel = (zs >= lo) & (zs < hi)
            rec[f"fixed_pool_sign_agreement {bname}"] = (rows[sel].mean(axis=0).tolist()
                                                         if sel.any() else None)
            rec[f"n_pools {bname}"] = int(sel.sum())
        out[f"beta={beta}"] = rec
    return out


def conditional_pool_curve(full, lp, rng, ks=(1, 2, 4, 6, 9, 12, 15, 18), n=4000):
    """The observed pool, resampled two ways: disjoint subsets of the fixed pool (what
    wrapper_subsets.py measured) against two INDEPENDENT k-samples with replacement from it."""
    nw = full.shape[1]
    fixed, indep = [], []
    for k in ks:
        fa, ia = 0, 0
        for _ in range(n):
            p = rng.permutation(nw)
            fa += np.sign(corr(lp, full[:, p[:k]].mean(1))) == np.sign(corr(lp, full[:, p[k:2 * k]].mean(1)))
            s1, s2 = rng.integers(0, nw, k), rng.integers(0, nw, k)
            ia += np.sign(corr(lp, full[:, s1].mean(1))) == np.sign(corr(lp, full[:, s2].mean(1)))
        fixed.append(fa / n)
        indep.append(ia / n)
    return {"k": list(ks), "fixed_pool_disjoint": fixed, "independent_with_replacement": indep}


def reproduction_check(name, z):
    """The cached real cells must reproduce the submitted ladder results before anything is built
    on them: the full-set statistic of ladder_curves.json, the components of pod_gtheory.json and
    the Spearman-Brown of pod_slope_reliability.json."""
    res = ROOT / "results"
    lc = json.loads((res / "ladder_curves.json").read_text(encoding="utf-8"))["ladders"][name]
    gt = json.loads((res / "pod_gtheory.json").read_text(encoding="utf-8"))["ladders"][name]
    sr = json.loads((res / "pod_slope_reliability.json").read_text(encoding="utf-8"))["ladders"][name]
    assert [list(p) for p in z["pairs"]] == lc["pairs"], "wrapper order differs from ladder_curves"
    d_full = float(np.abs(np.array(lc["statistic"]) - z["real_full"]).max())
    c = components_3way(z["real_blocks4"].astype(float))
    d_comp = max(abs(c[k] - v) / max(abs(v), 1e-12) for k, v in gt["components"].items()
                 if v > 1e-9)
    lp = np.log10(z["params_b"].astype(float))
    d_sb = abs(sb(lp, z["real_halves"].astype(float).transpose(1, 2, 0, 3)) - sr["spearman_brown"])
    ok = d_full < 2e-3 and d_comp < 1e-2 and d_sb < 2e-3
    if not ok:
        raise SystemExit(f"{name}: cached cells do not reproduce the stored results "
                         f"(statistic {d_full:.2e}, components {d_comp:.2e}, SB {d_sb:.2e})")
    return {"max_abs_diff_statistic": d_full, "max_rel_diff_components": d_comp,
            "abs_diff_spearman_brown": d_sb}


# ------------------------------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore")
    ladders = [n for n in LADDER_ORDER if (CELLS / f"{n}.npz").exists()]
    if not ladders:
        raise SystemExit("no cells under results/post_submission/cells; run ladder_audit_cells.py")
    print(f"ladders: {ladders}")
    avg_out, gs_out, null_out, floor_out, dis_out, pred = {}, {}, {}, {}, {}, {}
    pop = []
    for name in ladders:
        rng = np.random.default_rng(0)
        z = dict(np.load(CELLS / f"{name}.npz", allow_pickle=False))
        lp = np.log10(z["params_b"].astype(float))
        pop += [str(m) for m in z["models"]]
        n_e, n_d = len(z["eval_arms"]), len(z["deploy_arms"])
        ev_idx = [list(range(i * n_d, (i + 1) * n_d)) for i in range(n_e)]
        dp_idx = [list(range(j, n_e * n_d, n_d)) for j in range(n_d)]
        avg_out[name] = averaged_estimator(z, lp, rng)
        avg_out[name]["reproduces_stored_results"] = reproduction_check(name, z)
        gs_out[name] = crossed_gstudy(z, rng)
        null_out[name], floor_out[name] = null_pipeline(z, lp, ev_idx, dp_idx)
        dis_out[name] = arm_disjoint(z, lp, rng)
        dis_out[name]["observed_pool_resampling"] = conditional_pool_curve(
            z["real_full"].astype(float), lp, rng)
        a, g, nr = avg_out[name], gs_out[name], null_out[name]
        print(f"  {name}: r36 {a['r_36_average']:+.3f} items {a['ci_items']} arms {a['ci_arms']}"
              f" both {a['ci_items_and_arms']}  perm p1 {a['permutation_p_one_sided']:.3f}")
        print(f"     erho1 flat {g['erho_one_wrapper_full_items']['flat']:.3f} crossed "
              f"{g['erho_one_wrapper_full_items']['crossed']:.3f}  erho36 flat "
              f"{g['erho_36_wrappers_full_items']['flat']:.3f} crossed "
              f"{g['erho_36_wrappers_full_items']['crossed']:.3f}  k {g['k080']}  rel share "
              f"{g['relative_model_share']}")
        print(f"     null signflip SB {nr['signflip']['spearman_brown']['mean']:.3f} vs real "
              f"{nr['real']['spearman_brown']:.3f}; model share null "
              f"{nr['signflip']['model_share']['mean']:.3f} vs real {nr['real']['model_share']:.3f}",
              flush=True)

    # predictions of pod/PREDICTIONS.md section 4, scored on every rendering
    for name in ladders:
        if not name.startswith(("qwen3.5", "qwen3_")):
            continue
        nr, g = null_out[name]["real"], gs_out[name]
        pred[name] = {
            "replication_both_signs": nr["spans_both"],
            "components_model_below_mw": bool(nr["model_share"] < nr["mw_share"]),
            "components_mw_above_mi": bool(nr["mw_share"] > nr["mi_share"]),
            "reliability_erho_k1_below_080": bool(nr["erho_k1"] < TARGET),
            "model_share": nr["model_share"], "mw_share": nr["mw_share"], "mi_share": nr["mi_share"],
            "erho_k1": nr["erho_k1"],
            "bootstrap_p_model_below_mw": g["bootstrap"]["p_model_share_below_mw"],
            "bootstrap_ci_erho_k1_flat": g["bootstrap"]["erho1_flat"]["ci"],
        }
        held = [pred[name][k] for k in ("replication_both_signs", "components_model_below_mw",
                                         "components_mw_above_mi", "reliability_erho_k1_below_080")]
        pred[name]["n_held_of_4_directional_parts"] = int(sum(held))
        pred[name]["n_held_of_3_predictions"] = int(held[0] + (held[1] and held[2]) + held[3])

    conv = dict(pooling="validated for scoring and for the direction arrays "
                        "(Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                layers="ALL", scoring_array="results/post_submission/cells/*.npz, from "
                "results/pod_*/*.npz and results/gstudy via scripts/ladder_audit_cells.py",
                data_root="pod", model_population=sorted(set(pop)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    note = "post-submission audit; not part of the submitted manuscript or its recorded predictions"
    write_result(OUTDIR / "averaged_estimator.json", {"ladders": avg_out}, kind="measured",
                 statistic="r of the 36-wrapper-averaged folded max with log10 parameters", note=note,
                 **conv)
    write_result(OUTDIR / "crossed_gstudy.json", {"ladders": gs_out}, kind="measured",
                 statistic="folded max over layers, model x eval arm x deploy arm x item block",
                 note=note, **conv)
    write_result(OUTDIR / "null_directions.json", {"ladders": null_out, "floors": floor_out},
                 kind="measured", statistic="the pod_ladders pipeline on directions carrying no "
                 "label information", note=note, **conv)
    write_result(OUTDIR / "arm_disjoint.json",
                 {"ladders": dis_out,
                  "finite_pool_simulation": finite_pool_simulation(np.random.default_rng(0))},
                 kind="measured", statistic="sign and Kendall tau agreement of wrapper averages",
                 note=note, **conv)
    write_result(OUTDIR / "prediction_scoring.json", {"ladders": pred,
                 "source": "pod/PREDICTIONS.md section 4; the manuscript scored qwen3.5_nothink only"},
                 kind="measured", statistic="flat 3-way G-study and sign counts as pod_ladders",
                 note=note, **conv)
    print("wrote results/post_submission/{averaged_estimator,crossed_gstudy,null_directions,"
          "arm_disjoint,prediction_scoring}.json")


if __name__ == "__main__":
    main()
