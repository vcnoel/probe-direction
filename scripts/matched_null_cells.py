"""Cells for the matched-null audit: content-free directions that share the real directions' geometry.

scripts/ladder_audit_cells.py compared the real wrapper directions with isotropic and covariance
nulls. The real 36-wrapper average sat BELOW the isotropic floor, so those nulls differ from the
real directions in something other than label content. This script measures how they differ and
builds nulls that match the real directions on every geometric property except the label.

Per model and wrapper it records, for the real directions and for two draws of every null, the
properties that can move the statistic (folded max over all layers of |AUROC - 0.5|):

  norm profile        |d_l| per layer. AUROC is invariant to a positive rescaling at each layer, so
                      this cannot move the statistic; it is reported because the audit asked.
  cross-layer         lag-1 cosine of d_l with d_{l+1}, and the effective number of layers
                      n_eff = (sum lambda)^2 / sum lambda^2 of the L x L |cos| matrix. The maximum
                      over layers has about n_eff chances.
  framing subspace    share of |d_l|^2 in the top 1/5/20/100 principal components of the centred
                      framing activations at that layer (mean over 8 layers).
  scoring set         |cos| of d_l with the scoring set's class-mean difference (mean and max over
                      layers), the scoring-set variance along d_l relative to an isotropic
                      direction (median over layers), and the best-layer d' (outcome-adjacent;
                      reported as a diagnostic only).
  identity            |cos| with the real direction of the wrapper the null stands in for, against
                      |cos| between two independent draws of the same null (the subspace floor).

Nulls, N_DRAWS draws x 36 wrappers each (one draw = a full replacement set of 36 directions):

  iso           isotropic Gaussian, independent at every layer (as ladder_audit_cells).
  iso_rep       one isotropic Gaussian vector at every layer (as ladder_audit_cells).
  iso_xl        isotropic in the residual stream, but with the real direction's full L x L
                cross-layer cosine matrix: d = A e, A A^T = C_j, e iid N(0, I_D) per layer. Matches
                the number of effective layers exactly; does not live in the framing subspace.
  cov           random centred combination of the 1080 framing rows, one combination for all
                layers (as ladder_audit_cells; independent per wrapper).
  matched       the 36 real row-weight vectors rotated together by one random orthogonal map that
                preserves the energy in every dyadic band of the eigenspectrum of the layer-averaged
                framing Gram matrix (ranks 1, 2-3, 4-7, ... ). In whitened coordinates
                c = Lambda^{1/2} E^T w each band is rotated by a Haar matrix; w' = E Lambda^{-1/2} c'.
                One map per draw for all layers and all 36 wrappers, so the band energy profile,
                the Gram matrix among the 36 directions and the crossed arm structure are kept
                exactly (in the layer-averaged metric); cross-layer cosines are kept approximately.
  matched_wide  the same with wider bands (ranks 1-4, 5-16, 17-64, 65-256, 257-): less of the real
                direction survives the rotation, the energy profile is matched more coarsely.
  taskperm      within-task permutation of the 12 evaluation/deployment framing labels: for every
                task the 12 rows are shuffled, one shuffle per task per draw, shared by the 36
                wrappers and by every model of the ladder. The estimator, the task pairing and the
                crossed arm structure are kept; no wrapper keeps a consistent identity, so the
                direction is a random combination of framing differences.
  relabel       (exhaustive, not drawn) every unordered pair of the 12 framings is scored once;
                scripts/matched_null.py assembles all 461 non-identity 6/6 relabellings of the
                framings into pseudo-evaluation and pseudo-deployment arms from them.

Every null is scored on the full item set, the 4 seed-0 blocks and ALL 20 split halves of
pod_ladders. The real directions are rescored with the same fast scorer and must reproduce the
cells cached by ladder_audit_cells.py.

    EAS_ACTIVATION_ROOT=/path/to/root python scripts/matched_null_cells.py [ladder ...]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ladder_audit_cells import (band_label, band_layers, item_partitions, load_model,  # noqa: E402
                                wrapper_rows)
from pod_ladders import LADDERS, N_SPLITS  # noqa: E402

OUT = Path(os.environ.get("EAS_CELLS_DIR") or ROOT / "results" / "post_submission" / "cells")
N_DRAWS = int(os.environ.get("EAS_MATCHED_DRAWS", "100"))
N_DIAG = 2                     # draws per null whose directions are characterised
NULLS = ("iso", "iso_rep", "iso_xl", "cov", "matched", "matched_wide", "taskperm")
D_SPACE = ("iso", "iso_rep", "iso_xl")
BANDS = {"matched": [0, 1, 3, 7, 15, 31, 63, 127, 255, 511],
         "matched_wide": [0, 4, 16, 64, 256]}
TOPK = (1, 5, 20, 100)
SEED = 7
DRAW_CHUNK = 10                # draws generated at once for the residual-stream nulls
SORT_COLS = 8192               # columns sorted at once by one scorer thread
N_THREADS = int(os.environ.get("EAS_THREADS", "8"))


# ------------------------------------------------------------------------------------------------
# scorer: one sort per column, every item subset read off the same order
# ------------------------------------------------------------------------------------------------

def subset_list(y, blocks, halves):
    subs = [np.ones(y.size, bool)]
    for b in blocks[4]:
        m = np.zeros(y.size, bool)
        m[b] = True
        subs.append(m)
    for pair in halves:
        for h in pair:
            m = np.zeros(y.size, bool)
            m[h] = True
            subs.append(m)
    return subs


def peaks_multi(y, proj, subs):
    """proj [n_items, L, C] -> [n_subsets, C]: folded max over layers of |AUROC - 0.5| on every
    item subset. Ties have probability zero for continuous projections and are not averaged."""
    proj = np.ascontiguousarray(proj[:, band_layers(proj.shape[1]), :])
    n, L, C = proj.shape
    pos = np.stack([(y == 1) & s for s in subs]).astype(np.int16)      # [S, n]
    neg = np.stack([(y == 0) & s for s in subs]).astype(np.int16)
    denom = pos.sum(1).astype(float) * neg.sum(1)
    out = np.empty((len(subs), C))
    step = max(1, SORT_COLS // L)

    def chunk(s0):
        # numpy releases the GIL in argsort, take, cumsum and the reductions, so column chunks
        # run in parallel threads without copying the projections
        X = proj[:, :, s0:s0 + step]
        c = X.shape[2]
        order = np.argsort(X.reshape(n, L * c).T, axis=1)                 # [L*c, n]
        for k in range(len(subs)):
            no = np.take(neg[k], order)
            np.cumsum(no, axis=1, out=no)
            no *= np.take(pos[k], order)
            a = no.sum(axis=1, dtype=np.int64) / denom[k]
            a = a.reshape(L, c)
            out[k, s0:s0 + c] = np.abs(np.maximum(a, 1.0 - a) - 0.5).max(axis=0)

    with ThreadPoolExecutor(N_THREADS) as ex:
        list(ex.map(chunk, range(0, C, step)))
    return out


def split_cells(sc, n_halves):
    """[1 + 4 + 2 * n_halves, C] -> full [C], blocks4 [C, 4], halves [n_halves, 2, C]."""
    return sc[0], sc[1:5].T, sc[5:].reshape(n_halves, 2, -1)


# ------------------------------------------------------------------------------------------------
# direction properties
# ------------------------------------------------------------------------------------------------

def properties(U, sad, y, dmu_u, pcs, ref=None):
    """U [L, D, C] directions (any scale). Returns per-column property arrays."""
    L, D, C = U.shape
    nrm = np.linalg.norm(U, axis=1)                                        # [L, C]
    V = U / np.clip(nrm[:, None, :], 1e-30, None)
    adj = (V[:-1] * V[1:]).sum(axis=1).mean(axis=0)
    cm = np.abs(np.einsum("ldc,mdc->clm", V, V, optimize=True))
    ev = np.clip(np.linalg.eigvalsh(cm), 0, None)
    neff = ev.sum(1) ** 2 / (ev ** 2).sum(1)
    cdm = np.abs(np.einsum("ldc,ld->lc", V, dmu_u, optimize=True))
    vr, dpr = [], []
    for li in range(L):
        S = sad[li]
        P = S @ V[li]
        tr = S.var(axis=0, ddof=1).sum()
        vr.append(P.var(axis=0, ddof=1) / (tr / D))
        p1, p0 = P[y == 1], P[y == 0]
        dpr.append(np.abs(p1.mean(0) - p0.mean(0)) / np.sqrt(0.5 * (p1.var(0) + p0.var(0))))
    shares = {k: [] for k in TOPK}
    for li, Vt in pcs.items():
        c2 = (Vt @ V[li]) ** 2
        for k in TOPK:
            shares[k].append(c2[:k].sum(axis=0))
    prof = nrm / nrm.mean(axis=0, keepdims=True)
    out = {"adj_cos": adj, "n_eff_layers": neff, "cos_dmu_mean": cdm.mean(0),
           "cos_dmu_max": cdm.max(0), "scoring_var_ratio": np.median(np.array(vr), axis=0),
           "dprime_max": np.array(dpr).max(axis=0), "norm_profile": prof.mean(axis=1)}
    for k in TOPK:
        out[f"share_top{k}"] = np.mean(shares[k], axis=0)
    if ref is not None:
        R = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        j = np.arange(C) % R.shape[2]
        out["cos_real"] = np.abs((V * R[:, :, j]).sum(axis=1)).mean(axis=0)
        h = C // 2
        if h >= 1:
            out["cos_between_draws"] = np.abs((V[:, :, :h] * V[:, :, h:2 * h]).sum(axis=1)).mean(0)
    return out


# ------------------------------------------------------------------------------------------------
# null weights
# ------------------------------------------------------------------------------------------------

def haar_apply(cb, rng):
    """H @ cb for a Haar-random orthogonal H on the band, without forming H: with cb = U S V^T
    (thin, rank k), H U is a uniformly random orthonormal k-frame, drawn by QR of an n x k Gaussian.
    Same distribution as the full n x n rotation at O(n k^2) instead of O(n^3)."""
    n = cb.shape[0]
    if n == 1:
        return rng.choice([-1.0, 1.0]) * cb
    U, s, Vt = np.linalg.svd(cb, full_matrices=False)
    Q, R = np.linalg.qr(rng.standard_normal((n, s.size)))
    return (Q * np.sign(np.diag(R))) @ (s[:, None] * Vt)


def band_rotations(real_w, E, lam, edges, n_draws, rng):
    """Rotate the 36 real row-weight vectors together, band by band, in whitened coordinates."""
    r = lam.size
    edges = [e for e in edges if e < r] + [r]
    sq = np.sqrt(lam)
    c = sq[:, None] * (E.T @ real_w)                                        # [r, 36]
    out = np.empty((real_w.shape[0], n_draws * real_w.shape[1]), np.float32)
    for b in range(n_draws):
        c2 = np.empty_like(c)
        for lo, hi in zip(edges[:-1], edges[1:]):
            c2[lo:hi] = haar_apply(c[lo:hi], rng)
        out[:, b * c.shape[1]:(b + 1) * c.shape[1]] = E @ (c2 / sq[:, None])
    return out


def taskperm_rows(names, tasks, ev, dp, n_draws, rng):
    """n_draws row permutations, each shuffling the 12 eval/deploy rows within every task."""
    ed = np.isin(names, ev + dp)
    by_task = [np.flatnonzero(ed & (tasks == t)) for t in sorted(set(tasks[ed]))]
    perms = np.tile(np.arange(names.size), (n_draws, 1))
    for b in range(n_draws):
        for idx in by_task:
            perms[b, idx] = idx[rng.permutation(idx.size)]
    return perms


# ------------------------------------------------------------------------------------------------

def run_ladder(ladder, li_ladder):
    tmpl, params = LADDERS[ladder]
    models = list(params)
    meta, res = None, {}
    for mi, m in enumerate(models):
        t0 = time.time()
        sad, fr, names, tasks, y = load_model(ladder, m)
        pairs, ev, dp, rows, n_tasks = wrapper_rows(names, tasks)
        N = names.size
        if meta is None:
            blocks, halves = item_partitions(y)
            subs = subset_list(y, blocks, halves)
            real_w = np.zeros((N, len(pairs)), np.float32)
            for j, (fe, fd) in enumerate(pairs):
                real_w[rows[fe], j] = 1.0 / n_tasks
                real_w[rows[fd], j] = -1.0 / n_tasks
            arms = ev + dp
            pair_list = list(itertools.combinations(range(len(arms)), 2))
            pair_w = np.zeros((N, len(pair_list)), np.float32)
            for j, (a, b) in enumerate(pair_list):
                pair_w[rows[arms[a]], j] = 1.0 / n_tasks
                pair_w[rows[arms[b]], j] = -1.0 / n_tasks
            perms = taskperm_rows(names, tasks, ev, dp, N_DRAWS,
                                  np.random.default_rng([SEED, li_ladder, 99]))
            tp_w = np.concatenate([real_w[p] for p in perms], axis=1)
            meta = dict(y=y, pairs=pairs, names=names, tasks=tasks)
        else:
            assert np.array_equal(y, meta["y"]) and pairs == meta["pairs"]
            assert np.array_equal(names, meta["names"]) and np.array_equal(tasks, meta["tasks"])
        L, n, D = sad.shape
        C = len(pairs)
        rng = np.random.default_rng([SEED, li_ladder, mi])

        # per-layer Grams, layer-averaged centred Gram, scoring class-mean difference, PCs
        G = np.empty((L, n, N), np.float32)
        Kbar = np.zeros((N, N))
        dmu = np.empty((L, D), np.float32)
        Dreal = np.empty((L, D, C), np.float32)
        pcs = {}
        pc_layers = set(np.unique(np.linspace(0, L - 1, 8).round().astype(int)).tolist())
        rms = np.empty(L)
        for li in range(L):
            F = np.asarray(fr[:, li, :], dtype=np.float32)
            G[li] = sad[li] @ F.T
            Fc = F - F.mean(axis=0)
            K = (Fc @ Fc.T).astype(np.float64)
            Kbar += K / np.trace(K)
            dmu[li] = sad[li][y == 1].mean(0) - sad[li][y == 0].mean(0)
            Dreal[li] = F.T @ real_w
            rms[li] = np.sqrt((F ** 2).sum(1).mean())
            if li in pc_layers:
                lam_l, U_l = np.linalg.eigh(K)
                top = U_l[:, ::-1][:, :max(TOPK)]
                s = np.sqrt(np.clip(lam_l[::-1][:max(TOPK)], 1e-30, None))
                pcs[li] = ((Fc.T @ top.astype(np.float32)) / s.astype(np.float32)).T  # [k, D]
        Kbar /= L
        lam, E = np.linalg.eigh(Kbar)
        lam, E = lam[::-1], E[:, ::-1]
        keep = lam > lam[0] * 1e-10
        lam, E = lam[keep], E[:, keep]
        dmu_u = dmu / np.linalg.norm(dmu, axis=1, keepdims=True)

        r = {}
        # real directions: rescored, characterised
        Preal = np.ascontiguousarray((G @ real_w).transpose(1, 0, 2))
        r["real_full"], r["real_blocks4"], r["real_halves"] = split_cells(
            peaks_multi(y, Preal, subs), N_SPLITS)
        props = {"real": properties(Dreal, sad, y, dmu_u, pcs)}
        props["real"]["norm_rel_rms"] = (np.linalg.norm(Dreal, axis=1) / rms[:, None]).mean(1)
        cr = np.abs(np.einsum("ldc,lde->lce", Dreal / np.linalg.norm(Dreal, axis=1, keepdims=True),
                              Dreal / np.linalg.norm(Dreal, axis=1, keepdims=True),
                              optimize=True)).mean(0)
        props["real"]["cos_between_wrappers"] = cr[np.triu_indices(C, 1)]
        # K-bar band energy of the real directions (what the matched null preserves)
        cw = (np.sqrt(lam)[:, None] * (E.T @ real_w)) ** 2
        cw /= cw.sum(0, keepdims=True)
        props["real"]["kbar_band_energy"] = np.array(
            [cw[lo:hi].sum(0).mean() for lo, hi in
             zip(BANDS["matched"], BANDS["matched"][1:] + [lam.size])])
        props["real"]["kbar_spectrum_top20"] = lam[:20] / lam.sum()

        # relabel: all unordered framing pairs
        Pp = np.ascontiguousarray((G @ pair_w).transpose(1, 0, 2))
        r["pairs_full"], r["pairs_blocks4"], r["pairs_halves"] = split_cells(
            peaks_multi(y, Pp, subs), N_SPLITS)

        # cross-layer cosine square roots for iso_xl
        Vr = Dreal / np.linalg.norm(Dreal, axis=1, keepdims=True)
        Cmat = np.einsum("ldj,mdj->jlm", Vr, Vr, optimize=True).astype(np.float64)
        w_, Q_ = np.linalg.eigh(Cmat)
        Aroot = (Q_ * np.sqrt(np.clip(w_, 0, None))[:, None, :]) @ Q_.transpose(0, 2, 1)
        Aroot = Aroot.astype(np.float32)                                     # [36, L, L]

        for k in NULLS:
            if k in D_SPACE:
                P = np.empty((n, L, N_DRAWS * C), np.float32)
                diag = None
                for b0 in range(0, N_DRAWS, DRAW_CHUNK):
                    nb = min(DRAW_CHUNK, N_DRAWS - b0)
                    cols = slice(b0 * C, (b0 + nb) * C)
                    if k == "iso_rep":
                        e = rng.standard_normal((D, nb * C), dtype=np.float32)
                        for li in range(L):
                            P[:, li, cols] = sad[li] @ e
                        if diag is None:
                            diag = np.repeat(e[None, :, :N_DIAG * C], L, axis=0)
                        continue
                    e = rng.standard_normal((L, D, nb * C), dtype=np.float32)
                    if k == "iso_xl":
                        e = e.reshape(L, D * nb, C).transpose(2, 0, 1)        # [36, L, D*nb]
                        e = np.matmul(Aroot, e)
                        e = e.transpose(1, 2, 0).reshape(L, D, nb * C)
                    for li in range(L):
                        P[:, li, cols] = sad[li] @ e[li]
                    if diag is None:
                        diag = e[:, :, :N_DIAG * C].copy()
                    del e
            else:
                if k == "cov":
                    z = rng.standard_normal((N, N_DRAWS * C)).astype(np.float32)
                    W = (z - z.mean(axis=0)) / np.sqrt(N)
                elif k == "taskperm":
                    W = tp_w
                else:
                    W = band_rotations(real_w.astype(np.float64), E, lam, BANDS[k], N_DRAWS,
                                       rng).astype(np.float32)
                P = np.ascontiguousarray((G @ W).transpose(1, 0, 2))
                Wd = W[:, :N_DIAG * C]
                diag = np.stack([np.asarray(fr[:, li, :], np.float32).T @ Wd for li in range(L)])
            sc = peaks_multi(y, P, subs)
            del P
            full, b4, hv = split_cells(sc, N_SPLITS)
            r[f"{k}_full"] = full.reshape(N_DRAWS, C)
            r[f"{k}_blocks4"] = b4.reshape(N_DRAWS, C, 4)
            r[f"{k}_halves"] = hv.reshape(N_SPLITS, 2, N_DRAWS, C)
            props[k] = properties(diag, sad, y, dmu_u, pcs, ref=Dreal)
            del diag
        r["properties"] = np.array(json.dumps(
            {k: {kk: np.asarray(vv, float).tolist() for kk, vv in v.items()} for k, v in
             props.items()}))
        res[m] = r
        del sad, fr, G
        print(f"    {ladder} {m}: {L} layers, d={D}, rank {lam.size}  ({time.time() - t0:.0f}s)",
              flush=True)

    out = {k: np.stack([res[m][k] for m in models]) for k in res[models[0]]}
    out["models"] = np.array(models)
    out["params_b"] = np.array([params[m] for m in models])
    out["pairs"] = np.array(meta["pairs"])
    out["arms"] = np.array(arms)
    out["pair_index"] = np.array(pair_list)
    out["y"] = meta["y"]
    out["config"] = np.array(json.dumps({
        "n_draws": N_DRAWS, "n_splits": N_SPLITS, "nulls": list(NULLS), "bands": BANDS,
        "seed": SEED, "n_diag_draws": N_DIAG, "topk": list(TOPK),
        "statistic": band_label()["statistic"], "layer_band": band_label()["layer_band"],
        "activation_root_relative_paths": tmpl}))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"matched_{ladder}.npz", **out)
    print(f"  wrote {OUT.relative_to(ROOT).as_posix()}/matched_{ladder}.npz", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    only = sys.argv[1:]
    for i, ladder in enumerate(LADDERS):
        if only and ladder not in only:
            continue
        run_ladder(ladder, i)


if __name__ == "__main__":
    main()
