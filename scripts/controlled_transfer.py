"""Format-controlled transfer, vector averaging and framing-disjoint subsets.

Implements docs/controlled-transfer-registration.md (Q1 to Q5) as registered. Two stages:

  cells    per model, from the stored activations: every real, null and isotropic direction built
           on train tasks and scored on held-out tasks under every test wrapper (Q1 to Q3), and the
           paper's full-task directions projected on the paper's scoring set (Q4). The full-task
           projections must reproduce the October per-wrapper cells before anything is written.
  orient   Amendment 1 of the registration: framing means per split and band layer, for the
           orientation (A1) and alignment (A2) statistics.
  summary  the registered aggregations, the decision rules, Q5 and Amendment 1.

    EAS_ACTIVATION_ROOT=/path/to/root python scripts/controlled_transfer.py cells [ladder ...]
    EAS_ACTIVATION_ROOT=/path/to/root python scripts/controlled_transfer.py orient [ladder ...]
    python scripts/controlled_transfer.py summary
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import auc_cols, coefficients, components_3way, total  # noqa: E402
from pod_ladders import LADDERS, N_BLOCKS, peak  # noqa: E402
from result_io import write_result  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
CELLS_IN = ROOT / "results" / "post_submission" / "cells"
CELLS = ROOT / "results" / "post_submission" / "controlled"
OUT = ROOT / "results" / "post_submission" / "controlled_transfer.json"

N_SPLITS, SPLIT_SEED = 20, 0          # registered: 20 random 30/30 task splits, seed 0
N_ISO, ISO_SEED = 30, 1               # registered: 30 isotropic directions per layer, seed 1
KS = (1, 2, 3, 4, 6, 9, 12)           # registered group sizes for Q4
N_PART, PART_SEED = 50, 2             # 50 random partitions per k; the seed is fixed here
BAND = (0.25, 0.75)                   # registered relative-depth band for the signed statistic
REPRO_TOL = 2e-3                      # full-task cells must match the October cells to this


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def load_model(ladder, model):
    """Framing rows, task ids, scoring set and framing lists under the ladder's own convention."""
    tmpl, _ = LADDERS[ladder]
    if tmpl == "GSTUDY":
        d = ACT / "results" / "gstudy" / model
        with np.load(d / "pooled.npz", allow_pickle=False) as z:
            names = np.array([str(s) for s in z["framing_name"]])
            tasks = np.array([str(s) for s in z["framing_task"]])
            fam = json.loads(str(z["families"]))
            fr = z["framing"]
        with np.load(d / "sad_filtered_lf.npz", allow_pickle=False) as z2:
            sad, y = z2["sad"], z2["sad_label"].astype(int)
        ev = sorted(k for k, v in fam.items() if v == "eval")
        dp = sorted(k for k, v in fam.items() if v == "deploy")
    else:
        with np.load(ACT / tmpl.format(model), allow_pickle=False) as z:
            names = np.array([str(s) for s in z["framing_name"]])
            tasks = np.array([str(s) for s in z["framing_task"]])
            fr, sad, y = z["framing_validated"], z["sad_validated"], z["sad_label"].astype(int)
        ev = sorted({n for n in set(names) if n.startswith("eval_")})
        dp = sorted({n for n in set(names) if n.startswith("deploy_")})
    return names, tasks, fr, sad, y, ev, dp


def model_cells(ladder, model):
    names, tasks, fr, sad, y, ev, dp = load_model(ladder, model)
    frs = ev + dp
    task_ids = sorted(set(tasks[np.isin(names, frs)]))
    idx = np.zeros((len(frs), len(task_ids)), int)
    for i, f in enumerate(frs):
        rows = {t: r for r, t in enumerate(tasks) if names[r] == f}
        assert len(rows) == len(task_ids), f"{model}: {f} lacks a row for some task"
        idx[i] = [rows[t] for t in task_ids]
    n_t, L = len(task_ids), fr.shape[1]
    real = [(frs.index(e), frs.index(d)) for e, d in itertools.product(ev, dp)]   # pod_ladders order
    null = [(frs.index(a), frs.index(b)) for grp in (ev, dp) for a, b in itertools.combinations(grp, 2)]
    n_real, n_null = len(real), len(null)
    n_dir = n_real + n_null + N_ISO

    rng = np.random.default_rng(SPLIT_SEED)
    splits = [rng.permutation(n_t) for _ in range(N_SPLITS)]
    half = n_t // 2
    iso_rng = np.random.default_rng(ISO_SEED)

    proj = np.empty((N_SPLITS, len(frs), n_t - half, L, n_dir), np.float32)
    sadproj = np.empty((sad.shape[0], L, n_real), np.float64)
    for l in range(L):
        F = fr[idx, l, :].astype(np.float64)                      # [12 framings, tasks, d]
        mu = F.mean(1)
        sadproj[:, l, :] = sad[:, l, :].astype(np.float64) @ unit(
            np.stack([mu[e] - mu[d] for e, d in real])).T
        iso = unit(iso_rng.standard_normal((N_ISO, F.shape[2])))
        for s, perm in enumerate(splits):
            tr, te = perm[:half], perm[half:]
            m = F[:, tr, :].mean(1)
            D = np.vstack([unit(np.stack([m[a] - m[b] for a, b in real])),
                           unit(np.stack([m[a] - m[b] for a, b in null])), iso])
            proj[s, :, :, l, :] = F[:, te, :] @ D.T

    depth = np.arange(L) / (L - 1)
    band = (depth >= BAND[0]) & (depth <= BAND[1])
    n_te = n_t - half
    ylab = np.r_[np.ones(n_te, int), np.zeros(n_te, int)]
    fold = np.empty((N_SPLITS, n_real, n_dir), np.float32)
    sign = np.empty((N_SPLITS, n_real, n_dir), np.float32)
    for s in range(N_SPLITS):
        for k, (e2, d2) in enumerate(real):                       # test wrappers, same order
            X = np.concatenate([proj[s, e2], proj[s, d2]]).reshape(2 * n_te, L * n_dir)
            a = auc_cols(ylab, X).reshape(L, n_dir)
            fold[s, k] = np.abs(np.maximum(a, 1.0 - a) - 0.5).max(0)
            sign[s, k] = (a[band] - 0.5).mean(0)

    # validity of (direction, test wrapper): registered disjointness rules
    valid = np.zeros((n_real, n_dir), bool)
    for k, (e2, d2) in enumerate(real):
        for j, (e, d) in enumerate(real):
            valid[k, j] = e2 != e and d2 != d
        for i, (a, b) in enumerate(null):
            shared = {a, b} & {e2, d2}
            valid[k, n_real + i] = not shared
        valid[k, n_real + n_null:] = True

    full_peak = np.array([peak(y, sadproj[:, :, j]) for j in range(n_real)])
    return dict(fold=fold, sign=sign, valid=valid, sadproj=sadproj.astype(np.float32), y=y,
                full_peak=full_peak, L=L, n_real=n_real, n_null=n_null, n_iso=N_ISO,
                framings=np.array(frs), real=np.array(real), null=np.array(null))


def stage_cells(ladders):
    CELLS.mkdir(parents=True, exist_ok=True)
    for name in ladders:
        tmpl, params = LADDERS[name]
        with np.load(CELLS_IN / f"matched_{name}.npz", allow_pickle=False) as z:
            ref_models = [str(m) for m in z["models"]]
            ref_full = z["real_full"]
        assert ref_models == list(params), f"{name}: model order differs from the October cells"
        for i, model in enumerate(params):
            out = CELLS / f"{name}__{model}.npz"
            if out.exists():
                print(f"  have {out.name}", flush=True)
                continue
            c = model_cells(name, model)
            diff = float(np.max(np.abs(c["full_peak"] - ref_full[i])))
            if diff > REPRO_TOL:
                raise SystemExit(f"{name}/{model}: full-task cells differ from October by {diff:.4g}")
            np.savez_compressed(out, repro_maxdiff=diff, **c)
            print(f"  wrote {out.name}  (October cells reproduced to {diff:.1e})", flush=True)


def ladder_summary(name):
    tmpl, params = LADDERS[name]
    models = list(params)
    lp = np.array([np.log10(params[m]) for m in models])
    C = [dict(np.load(CELLS / f"{name}__{m}.npz", allow_pickle=False)) for m in models]
    n_real, n_null = int(C[0]["n_real"]), int(C[0]["n_null"])
    sl_real, sl_null, sl_iso = slice(0, n_real), slice(n_real, n_real + n_null), slice(n_real + n_null, None)

    def controlled(c, key):
        v = np.where(c["valid"][None], c[key], np.nan)            # [split, test wrapper, dir]
        return np.nanmedian(v.reshape(-1, v.shape[2]), axis=0)

    q1, cf, cs, nf, ns = [], [], [], [], []
    for m, c in zip(models, C):
        f, s = controlled(c, "fold"), controlled(c, "sign")
        rf, rs = float(np.median(f[sl_real])), float(np.median(s[sl_real]))
        tf, ts = float(np.percentile(f[sl_null], 95)), float(np.percentile(s[sl_null], 95))
        diag = np.array([c["fold"][:, j, j] for j in range(n_real)]).mean()
        q1.append({"model": m, "real_fold": rf, "null_fold_p95": tf, "null_fold_median":
                   float(np.median(f[sl_null])), "iso_fold_median": float(np.median(f[sl_iso])),
                   "real_sign": rs, "null_sign_p95": ts, "null_sign_median": float(np.median(s[sl_null])),
                   "iso_sign_median": float(np.median(s[sl_iso])),
                   "share_real_above_null_p95_fold": float(np.mean(f[sl_real] > tf)),
                   "share_real_above_null_p95_sign": float(np.mean(s[sl_real] > ts)),
                   "same_wrapper_fold_mean": float(diag),
                   "pass": bool(rf > tf and rs > ts)})
        cf.append(f[sl_real]); cs.append(s[sl_real]); nf.append(f[sl_null]); ns.append(s[sl_null])
    cf, cs, nf, ns = map(np.array, (cf, cs, nf, ns))              # [model, direction]
    need = int(np.ceil(0.75 * len(models)))
    q1_holds = sum(r["pass"] for r in q1) >= need

    def rvec(mat):
        return np.array([np.corrcoef(lp, mat[:, j])[0, 1] for j in range(mat.shape[1])])

    real_pairs = [tuple(p) for p in C[0]["real"]]
    with np.load(CELLS_IN / f"matched_{name}.npz", allow_pickle=False) as z:
        sad_full = z["real_full"]
        taskperm = z["taskperm_full"]
    by_e, by_d = {}, {}
    for j, (e, d) in enumerate(real_pairs):
        by_e.setdefault(e, []).append(j)
        by_d.setdefault(d, []).append(j)

    def sign_summary(r):
        return {"r_min": float(r.min()), "r_max": float(r.max()), "n_positive": int((r > 0).sum()),
                "n_negative": int((r < 0).sum()),
                "eval_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_e.values())),
                "deploy_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_d.values()))}

    r_cf, r_cs, r_sad = rvec(cf), rvec(cs), rvec(sad_full)
    q2 = {"controlled_fold": sign_summary(r_cf), "controlled_sign": sign_summary(r_cs),
          "paper_scoring_set": sign_summary(r_sad),
          "null_controlled_fold": {"r_min": float(rvec(nf).min()), "r_max": float(rvec(nf).max()),
                                   "n_positive": int((rvec(nf) > 0).sum()), "n": int(nf.shape[1])},
          "corr_real_r_controlled_vs_paper_set": float(np.corrcoef(r_cf, r_sad)[0, 1])}
    q2["holds"] = bool(q1_holds and q2["controlled_fold"]["n_positive"] >= 6
                       and q2["controlled_fold"]["n_negative"] >= 6)

    # Q3: model x train wrapper x test wrapper
    x = np.stack([c["fold"][:, :, sl_real].mean(0).T for c in C])  # [model, train j, test k]
    def gstudy(arr):
        comp = components_3way(arr)
        tot = total(comp)
        k080 = next((k for k in range(1, 100001) if coefficients(comp, k, arr.shape[2])[0] >= 0.80), None)
        return {"shares": {k: float(v / tot) for k, v in comp.items() if not k.startswith("_")},
                "truncated_at_zero": list(comp["_truncated"]),
                "erho_single": float(coefficients(comp, 1, 1)[0]), "k_for_080_at_n_test": k080}
    xs = x.copy()
    vmask = C[0]["valid"][:, sl_real].T                          # [train j, test k]
    for mi in range(xs.shape[0]):
        for j in range(n_real):
            xs[mi, j, ~vmask[j]] = x[mi, j, vmask[j]].mean()
    q3 = {"all_cells": gstudy(x), "shared_framing_cells_imputed": gstudy(xs),
          "mapping": "s_m model, s_t train wrapper, s_h test wrapper, s_mt model x train wrapper"}

    # Q4: vector averaging vs scalar averaging on the paper's scoring set
    y = C[0]["y"]
    rng0 = np.random.default_rng(0)                               # pod_ladders blocks
    pos, neg = rng0.permutation(np.flatnonzero(y == 1)), rng0.permutation(np.flatnonzero(y == 0))
    blocks = [np.concatenate([p, q]) for p, q in zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
    SP = [c["sadproj"].astype(np.float64) for c in C]
    xw = np.array([[[peak(y[b], sp[b][:, :, j]) for b in blocks] for j in range(n_real)] for sp in SP])
    cw = components_3way(xw)
    prng = np.random.default_rng(PART_SEED)
    q4 = []
    for k in KS:
        vals = []
        for _ in range(1 if k == 1 else N_PART):
            groups = np.arange(n_real) if k == 1 else prng.permutation(n_real)
            groups = groups.reshape(n_real // k, k)
            xg = np.array([[[peak(y[b], sp[b][:, :, g].mean(2)) for b in blocks] for g in groups] for sp in SP])
            vals.append(coefficients(components_3way(xg), 1, 1)[0])
        q4.append({"k": k, "erho_vector_median": float(np.nanmedian(vals)),
                   "erho_vector_q10_q90": [float(np.nanpercentile(vals, 10)), float(np.nanpercentile(vals, 90))],
                   "erho_scalar": float(coefficients(cw, k, 1)[0])})
    vec36 = np.array([peak(y, sp.mean(2)) for sp in SP])
    sc36 = sad_full.mean(1)
    q4 = {"curve": q4, "vector_rises_faster": bool(all(r["erho_vector_median"] > r["erho_scalar"]
                                                        for r in q4 if r["k"] >= 2)),
          "r_36_vector_average": float(np.corrcoef(lp, vec36)[0, 1]),
          "r_36_scalar_average": float(np.corrcoef(lp, sc36)[0, 1])}

    # Q5: framing-disjoint 3 x 3 subsets, real vs taskperm null
    ev = sorted({e for e, _ in real_pairs}); dp = sorted({d for _, d in real_pairs})
    parts = [(EA, DA) for EA in itertools.combinations(ev, 3) if ev[0] in EA
             for DA in itertools.combinations(dp, 3)]
    def agreement(mat):
        rho, same = [], []
        for EA, DA in parts:
            A = [j for j, (e, d) in enumerate(real_pairs) if e in EA and d in DA]
            B = [j for j, (e, d) in enumerate(real_pairs) if e not in EA and d not in DA]
            sa, sb = mat[:, A].mean(1), mat[:, B].mean(1)
            rho.append(spearmanr(sa, sb)[0])
            same.append(np.sign(np.corrcoef(lp, sa)[0, 1]) == np.sign(np.corrcoef(lp, sb)[0, 1]))
        return np.array(rho), np.array(same)
    rho_r, same_r = agreement(sad_full)
    null_med = np.array([np.nanmedian(agreement(taskperm[:, t, :])[0]) for t in range(taskperm.shape[1])])
    q5 = {"n_partitions": len(parts), "real_spearman_median": float(np.nanmedian(rho_r)),
          "real_spearman_q10_q90": [float(np.nanpercentile(rho_r, 10)), float(np.nanpercentile(rho_r, 90))],
          "real_same_scaling_sign_share": float(same_r.mean()),
          "null_median_of_medians": float(np.nanmedian(null_med)),
          "null_p95_of_medians": float(np.nanpercentile(null_med, 95)),
          "exceeds_null": bool(np.nanmedian(rho_r) > np.nanpercentile(null_med, 95))}

    return {"models": models, "params_b": [params[m] for m in models],
            "repro_maxdiff": float(max(float(c["repro_maxdiff"]) for c in C)),
            "q1": {"per_model": q1, "models_needed": need, "holds": bool(q1_holds)},
            "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def model_orient(ladder, model):
    """Amendment 1: framing means per split and band layer, reduced to what A1 and A2 need.

    G[s, l, w, p]   held-out mean of framing w projected on unit(m_a - m_b), p = (a, b), a < b
    COS[s, l, p, q] cos(unit(m_a - m_b) on train tasks, unit(t_c - t_d) on test tasks), q = (c, d)
    """
    names, tasks, fr, sad, y, ev, dp = load_model(ladder, model)
    frs = ev + dp
    task_ids = sorted(set(tasks[np.isin(names, frs)]))
    idx = np.array([[{t: r for r, t in enumerate(tasks) if names[r] == f}[t] for t in task_ids]
                    for f in frs])
    L = fr.shape[1]
    band = np.flatnonzero((np.arange(L) / (L - 1) >= BAND[0]) & (np.arange(L) / (L - 1) <= BAND[1]))
    pairs = list(itertools.combinations(range(len(frs)), 2))
    a_i, b_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    rng = np.random.default_rng(SPLIT_SEED)                      # the registered splits
    splits = [rng.permutation(len(task_ids)) for _ in range(N_SPLITS)]
    half = len(task_ids) // 2
    G = np.empty((N_SPLITS, band.size, len(frs), len(pairs)), np.float32)
    COS = np.empty((N_SPLITS, band.size, len(pairs), len(pairs)), np.float32)
    for li, l in enumerate(band):
        F = fr[idx, l, :].astype(np.float64)
        for s, perm in enumerate(splits):
            m = F[:, perm[:half], :].mean(1)
            t = F[:, perm[half:], :].mean(1)
            dtr, dte = unit(m[a_i] - m[b_i]), unit(t[a_i] - t[b_i])
            G[s, li] = t @ dtr.T
            COS[s, li] = dtr @ dte.T
    return dict(G=G, COS=COS, framings=np.array(frs), n_eval=len(ev), pairs=np.array(pairs),
                band=band, L=L)


def stage_orient(ladders):
    CELLS.mkdir(parents=True, exist_ok=True)
    for name in ladders:
        for model in LADDERS[name][1]:
            out = CELLS / f"orient_{name}__{model}.npz"
            if out.exists():
                print(f"  have {out.name}", flush=True)
                continue
            np.savez_compressed(out, **model_orient(name, model))
            print(f"  wrote {out.name}", flush=True)


def crossing_partitions(n_eval, n_fr):
    """All 6/6 partitions with 3 evaluation and 3 deployment framings per group, each listed once."""
    ev, dp = list(range(n_eval)), list(range(n_eval, n_fr))
    out = []
    for EA in itertools.combinations(ev, n_eval // 2):
        if ev[0] not in EA:
            continue
        for DA in itertools.combinations(dp, len(dp) // 2):
            g1 = list(EA) + list(DA)
            out.append((g1, [f for f in range(n_fr) if f not in g1]))
    return out


def orient_summary(name):
    tmpl, params = LADDERS[name]
    models = list(params)
    lp = np.array([np.log10(params[m]) for m in models])
    per_model, wrap_excess, X = [], [], []
    for model in models:
        z = np.load(CELLS / f"orient_{name}__{model}.npz", allow_pickle=False)
        G, COS, pairs, n_eval = z["G"], z["COS"], [tuple(p) for p in z["pairs"]], int(z["n_eval"])
        n_fr = G.shape[2]
        pid = {p: i for i, p in enumerate(pairs)}
        # positive fraction of (G[w1] - G[w2]) over splits and band layers, per unordered direction
        diff = G[:, :, :, None, :] - G[:, :, None, :, :]          # [s, l, w1, w2, p]
        pf = (diff > 0).mean(axis=(0, 1))                          # [w1, w2, p]
        A = COS.mean(axis=(0, 1))                                  # [p, q]

        def ori(u, v, u2, v2):
            return pf[u2, v2, pid[(u, v)]] if u < v else pf[v2, u2, pid[(v, u)]]

        def ali(u, v, u2, v2):
            sp = 1.0 if u < v else -1.0
            sq = 1.0 if u2 < v2 else -1.0
            return sp * sq * A[pid[tuple(sorted((u, v)))], pid[tuple(sorted((u2, v2)))]]

        ev, dp = range(n_eval), range(n_eval, n_fr)
        real = [(e, d) for e in ev for d in dp]
        r_ori = [ori(e, d, e2, d2) for e, d in real for e2, d2 in real if e2 != e and d2 != d]
        r_ali_w = [np.median([ali(e, d, e2, d2) for e2, d2 in real if e2 != e and d2 != d]) for e, d in real]
        r_ali = [ali(e, d, e2, d2) for e, d in real for e2, d2 in real if e2 != e and d2 != d]
        b_ori, b_ali = [], []
        for g1, g2 in crossing_partitions(n_eval, n_fr):
            dirs = [(a, b) for a in g1 for b in g2]
            b_ori.append(np.mean([ori(a, b, a2, b2) for a, b in dirs for a2, b2 in dirs if a2 != a and b2 != b]))
            b_ali.append(np.median([ali(a, b, a2, b2) for a, b in dirs for a2, b2 in dirs if a2 != a and b2 != b]))
        Ro, p95 = float(np.mean(r_ori)), float(np.percentile(b_ori, 95))
        base = float(np.median(b_ali))
        per_model.append({"model": model, "real_orientation": Ro, "baseline_orientation_median":
                          float(np.median(b_ori)), "baseline_orientation_p95": p95,
                          "real_alignment_median": float(np.median(r_ali)),
                          "baseline_alignment_median": base,
                          "excess_alignment": float(np.median(r_ali) - base), "pass": bool(Ro > p95)})
        wrap_excess.append(np.array(r_ali_w) - base)
        X.append(np.array([[ali(e, d, e2, d2) for e2, d2 in real] for e, d in real]))
    need = int(np.ceil(0.75 * len(models)))
    a1 = sum(r["pass"] for r in per_model) >= need
    W = np.array(wrap_excess)                                      # [model, wrapper]
    r = np.array([np.corrcoef(lp, W[:, j])[0, 1] for j in range(W.shape[1])])
    by_e = [list(range(i * 6, i * 6 + 6)) for i in range(6)]
    by_d = [list(range(i, 36, 6)) for i in range(6)]
    q2 = {"r_min": float(r.min()), "r_max": float(r.max()), "n_positive": int((r > 0).sum()),
          "n_negative": int((r < 0).sum()),
          "eval_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_e)),
          "deploy_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_d)),
          "r_mean_excess_alignment_vs_size": float(np.corrcoef(lp, W.mean(1))[0, 1])}
    q2["holds"] = bool(a1 and q2["n_positive"] >= 6 and q2["n_negative"] >= 6)
    x = np.stack(X)                                                # [model, train j, test k]
    xs = x.copy()
    real_pairs = [(e, d) for e in range(6) for d in range(6)]
    for j, (e, d) in enumerate(real_pairs):
        shared = np.array([e2 == e or d2 == d for e2, d2 in real_pairs])
        xs[:, j, shared] = x[:, j, ~shared].mean(1, keepdims=True)
    def gstudy(arr):
        comp = components_3way(arr)
        tot = total(comp)
        k080 = next((k for k in range(1, 100001) if coefficients(comp, k, arr.shape[2])[0] >= 0.80), None)
        return {"shares": {k: float(v / tot) for k, v in comp.items() if not k.startswith("_")},
                "truncated_at_zero": list(comp["_truncated"]),
                "erho_single": float(coefficients(comp, 1, 1)[0]), "k_for_080_at_n_test": k080}
    return {"a1": {"per_model": per_model, "models_needed": need, "holds": bool(a1)},
            "q2_amended": q2, "q3_amended": {"all_cells": gstudy(x), "shared_framing_cells_imputed": gstudy(xs)}}


def amendment2(name):
    """B1 null for the sign split and B2 split-half reliability of the per-wrapper sign."""
    tmpl, params = LADDERS[name]
    models = list(params)
    lp = np.array([np.log10(params[m]) for m in models])
    Z = [np.load(CELLS / f"orient_{name}__{m}.npz", allow_pickle=False) for m in models]
    pairs = [tuple(p) for p in Z[0]["pairs"]]
    pid = {p: i for i, p in enumerate(pairs)}
    n_eval, n_fr = int(Z[0]["n_eval"]), Z[0]["G"].shape[2]
    A = np.stack([z["COS"].mean(axis=(0, 1)) for z in Z])         # [model, p, q]

    def ali(u, v, u2, v2):
        sp = 1.0 if u < v else -1.0
        sq = 1.0 if u2 < v2 else -1.0
        return sp * sq * A[:, pid[tuple(sorted((u, v)))], pid[tuple(sorted((u2, v2)))]]

    def direction_r(dirs, tests, base):
        r = []
        for a, b in dirs:
            ok = [(a2, b2) for a2, b2 in tests if a2 != a and b2 != b and a2 != b and b2 != a]
            if len(ok) < 2:
                r.append(np.nan)
                continue
            exc = np.median(np.stack([ali(a, b, a2, b2) for a2, b2 in ok]), axis=0) - base
            r.append(np.corrcoef(lp, exc)[0, 1])
        return np.array(r)

    ev, dp = list(range(n_eval)), list(range(n_eval, n_fr))
    real = [(e, d) for e in ev for d in dp]
    cross = crossing_partitions(n_eval, n_fr)
    base = np.array([np.median([ali(a, b, a2, b2)[mi] for g1, g2 in cross for a in g1 for b in g2
                                for a2 in g1 for b2 in g2 if a2 != a and b2 != b])
                     for mi in range(len(models))])                # model baseline medians
    r_real = direction_r(real, real, base)
    null_minor, null_sd = [], []
    for g1, g2 in cross:
        dirs = [(a, b) for a in g1 for b in g2]
        rn = direction_r(dirs, dirs, base)
        null_minor.append(min((rn > 0).sum(), (rn < 0).sum()))
        null_sd.append(np.nanstd(rn))
    minor = min((r_real > 0).sum(), (r_real < 0).sum())

    def midrank(v, dist):
        dist = np.asarray(dist, float)
        return float(100 * ((dist < v).mean() + 0.5 * (dist == v).mean()))

    def sb(r):
        return 2 * r / (1 + r) if r > -1 else np.nan

    def halves(dirs, ev_set, dp_set):
        rel = []
        for EA in itertools.combinations(ev_set, 3):
            if ev_set[0] not in EA:
                continue
            for DA in itertools.combinations(dp_set, 3):
                EB = [e for e in ev_set if e not in EA]
                DB = [d for d in dp_set if d not in DA]
                ta = [(e, d) for e in EA for d in DA]
                tb = [(e, d) for e in EB for d in DB]
                ra, rb = direction_r(dirs, ta, base), direction_r(dirs, tb, base)
                ok = ~np.isnan(ra) & ~np.isnan(rb)
                if ok.sum() >= 3:
                    rel.append(sb(np.corrcoef(ra[ok], rb[ok])[0, 1]))
        return float(np.nanmedian(rel)) if rel else float("nan")

    rel_real = halves(real, ev, dp)
    rel_null = []
    for g1, g2 in cross:
        dirs = [(a, b) for a in g1 for b in g2]
        rel_null.append(halves(dirs, sorted(g1), sorted(g2)))
    return {"b1": {"real_minority_sign_count": int(minor), "real_sd_r": float(np.nanstd(r_real)),
                   "minority_percentile_in_null": midrank(minor, null_minor),
                   "sd_percentile_in_null": midrank(float(np.nanstd(r_real)), null_sd),
                   "null_minority_median": float(np.median(null_minor)),
                   "null_sd_median": float(np.median(null_sd))},
            "b2": {"real_reliability": rel_real, "null_median": float(np.nanmedian(rel_null)),
                   "null_p95": float(np.nanpercentile(rel_null, 95)),
                   "reliable_beyond_null": bool(rel_real > np.nanpercentile(rel_null, 95))}}


def stage_summary():
    out = {}
    for name in LADDERS:
        if all((CELLS / f"{name}__{m}.npz").exists() for m in LADDERS[name][1]):
            out[name] = ladder_summary(name)
            s = out[name]
            if all((CELLS / f"orient_{name}__{m}.npz").exists() for m in LADDERS[name][1]):
                s["amendment_1"] = orient_summary(name)
                s["amendment_2"] = amendment2(name)
                b = s["amendment_2"]
                print(f"{name:16s} B1 minority {b['b1']['real_minority_sign_count']} (null pct "
                      f"{b['b1']['minority_percentile_in_null']:.0f}), B2 reliability "
                      f"{b['b2']['real_reliability']:.2f} vs null p95 {b['b2']['null_p95']:.2f}")
                a = s["amendment_1"]
                print(f"{name:16s} A1 {'holds' if a['a1']['holds'] else 'fails'} "
                      f"({sum(r['pass'] for r in a['a1']['per_model'])}/{len(s['models'])})  "
                      f"Q2 amended +{a['q2_amended']['n_positive']}/-{a['q2_amended']['n_negative']}")
            print(f"{name:16s} Q1 {'holds' if s['q1']['holds'] else 'fails'} "
                  f"({sum(r['pass'] for r in s['q1']['per_model'])}/{len(s['models'])})  "
                  f"Q2 fold +{s['q2']['controlled_fold']['n_positive']}/-{s['q2']['controlled_fold']['n_negative']}  "
                  f"Q4 vector faster {s['q4']['vector_rises_faster']}  Q5 exceeds null {s['q5']['exceeds_null']}")
        else:
            print(f"{name:16s} cells incomplete, skipped")
    mn = json.loads((ROOT / "results" / "post_submission" / "matched_null.json").read_text())
    cos = {lad: {null: v["properties"][null]["cos_real"]["mean_over_models"]
                 for null in v.get("properties", {}) if null != "real"
                 and "cos_real" in v["properties"][null]}
           for lad, v in mn.get("ladders", {}).items()}
    pop = [m for n in out for m in out[n]["models"]]
    write_result(OUT, {"ladders": out, "null_cosines_from_matched_null": cos,
                       "registration": "docs/controlled-transfer-registration.md"},
                 kind="measured",
                 statistic="Q1-Q3: fold = max over layers |AUROC-0.5| and sign = mean over relative "
                           "depth 0.25-0.75 of (AUROC-0.5), on held-out tasks under disjoint test "
                           "wrappers; Q4-Q5: the paper's statistic on its scoring set",
                 note="post-submission analysis registered before computation; not part of the "
                      "submitted manuscript",
                 pooling="validated for directions and scoring (Qwen2.5: lf scoring, all-token "
                         "directions, as pod_ladders)",
                 layers="ALL for fold and Q4-Q5; relative depth 0.25-0.75 for sign",
                 scoring_array="held-out framing rows of results/pod_*/*.npz and results/gstudy "
                               "(Q1-Q3); results/post_submission/cells/matched_*.npz (Q5)",
                 data_root="pod", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) < 2 or sys.argv[1] not in ("cells", "orient", "summary"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "cells":
        stage_cells(sys.argv[2:] or list(LADDERS))
    elif sys.argv[1] == "orient":
        stage_orient(sys.argv[2:] or list(LADDERS))
    else:
        stage_summary()
