"""Cell statistics for the post-submission audit of the crossed wrapper design, on every ladder.

scripts/pod_ladders.py scores each (model, wrapper) cell once on the full item set, once on four
item blocks and on twenty split halves, for the 36 real directions only. The audit needs the same
cells under resampling and under directions that carry no label information, so this computes them
once, model by model, and caches them for scripts/ladder_audit.py.

Every direction used here is a linear combination of the model's 1080 framing activations (except
the isotropic null), so the projection of the scoring set onto it is a product with the Gram matrix
G_l = S_l F_l^T (400 x 1080 per layer). AUROC is invariant to a positive rescaling of the direction
at each layer, so the per-layer normalisation of pod_ladders.unit is immaterial and the real cells
reproduce results/ladder_curves.json and results/pod_gtheory.json (checked by ladder_audit.py).

Cells computed, statistic = folded maximum over ALL layers of |AUROC - 0.5| (pod_ladders.peak):

  real     36 wrappers: full set; the 4 seed-0 label-stratified blocks of pod_ladders; the 20
           seed-0 split halves of pod_ladders (same RNG sequence, so the Spearman-Brown value is
           reproduced); partitions into 2 and 8 blocks from the same permutation; N_BOOT item
           bootstrap replicates (items resampled with replacement within label, identical indices
           for every model of the ladder), each scored on the full resample and on its 4 blocks.
  nulls    N_NULL draws x 36 wrappers per null type, scored on the full set, the same 4 blocks
           and the first N_SPLITS_NULL of the same split halves:
             signflip  paired label permutation: d = mean_t s_t (a_et - a_dt), s_t = +-1 per task.
                       The estimator, the pairing and the two framings are kept; only the labels
                       that give the direction meaning are destroyed. Same signs for every model.
             permute   unpaired label permutation of the wrapper's 180 framing rows (90 vs 90).
             cov       Gaussian direction with the covariance of the model's 1080 framing
                       activations (a random centred combination of the framing rows, one
                       combination for all layers, as a real direction is).
             iso       isotropic Gaussian direction drawn independently at every layer (the most
                       tries in the maximum, so the highest floor).
             iso_rep   one isotropic Gaussian vector reused at every layer (the fewest tries, so
                       the lowest floor). iso and iso_rep bracket a content-free floor.
           For each null the median |cos| with the real direction of the wrapper it stands in for is
           stored (cos_<null>). The paired sign flip turns out to be almost the real direction
           (|cos| ~ 0.99): within a task the eval-minus-deploy difference is dominated by the
           common framing shift, so flipping signs only rescales it. It is kept as a diagnostic of
           that degeneracy, not as a content-free null.

The cache is results/post_submission/cells/<ladder>.npz (activation-derived, gitignored like every
other .npz). Activations are read from $EAS_ACTIVATION_ROOT (default: the repository root), under
the same relative paths pod_ladders.LADDERS names.

    EAS_ACTIVATION_ROOT=/path/to/root python scripts/ladder_audit_cells.py [ladder ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import auc_cols  # noqa: E402
from pod_ladders import LADDERS, N_BLOCKS, N_SPLITS  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
OUT = Path(os.environ.get("EAS_CELLS_DIR") or ROOT / "results" / "post_submission" / "cells")
# Layer band for the statistic (docs/interior-band-registration.md). Unset: all layers, as submitted.
_BAND = os.environ.get("EAS_LAYER_BAND")


def band_layers(n_layers):
    """Layer indices the statistic is maximised over: all, or relative depth in the registered band."""
    if not _BAND:
        return slice(None)
    lo, hi = map(float, _BAND.split(","))
    depth = np.arange(n_layers) / (n_layers - 1)
    return np.flatnonzero((depth >= lo) & (depth <= hi))


def band_label():
    """What the statistic is maximised over, for config and conventions records."""
    if not _BAND:
        return {"statistic": "folded max over all layers of |AUROC - 0.5|", "layers": "ALL",
                "layer_band": None}
    lo, hi = map(float, _BAND.split(","))
    where = (f"hidden states at relative depth {lo:g} to {hi:g} (index / (number of hidden "
             f"states - 1); hidden state 0, the embedding output, counts in the depth)")
    return {"statistic": f"folded max of |AUROC - 0.5| over {where}", "layers": where,
            "layer_band": [lo, hi]}

N_BOOT = 200
N_NULL = 20
N_SPLITS_NULL = 10
NULLS = ("signflip", "permute", "cov", "iso", "iso_rep")
SEED_BOOT, SEED_NULL = 1, 2
CHUNK_COLS = 4096


def item_partitions(y):
    """pod_ladders' seed-0 blocks and split halves, drawn in its exact RNG order."""
    rng = np.random.default_rng(0)
    pos, neg = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0))
    blocks = {nb: [np.concatenate([p, q]) for p, q in
                   zip(np.array_split(pos, nb), np.array_split(neg, nb))] for nb in (2, 4, 8)}
    assert N_BLOCKS == 4
    halves = []
    for _ in range(N_SPLITS):
        pa, na = rng.permutation(pos), rng.permutation(neg)
        halves.append((np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]]),
                       np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])))
    return blocks, halves


def bootstrap_indices(y):
    rng = np.random.default_rng(SEED_BOOT)
    pos0, neg0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    reps = []
    for _ in range(N_BOOT):
        pos = rng.choice(pos0, size=pos0.size, replace=True)
        neg = rng.choice(neg0, size=neg0.size, replace=True)
        pp, nn = rng.permutation(pos), rng.permutation(neg)
        blocks = [np.concatenate([p, q]) for p, q in
                  zip(np.array_split(pp, N_BLOCKS), np.array_split(nn, N_BLOCKS))]
        reps.append((np.concatenate([pos, neg]), blocks))
    return reps


def peaks(y, proj):
    """proj [n_items, n_layers, n_cols] -> folded max over layers per column."""
    proj = proj[:, band_layers(proj.shape[1]), :]
    n, L, c = proj.shape
    out = np.empty(c)
    step = max(1, CHUNK_COLS // L)          # bounds the rank arrays on a shared machine
    for s in range(0, c, step):
        p = proj[:, :, s:s + step]
        a = auc_cols(y, p.reshape(n, -1)).reshape(L, p.shape[2])
        out[s:s + step] = np.nanmax(np.abs(np.maximum(a, 1.0 - a) - 0.5), axis=0)
    return out


def load_model(ladder, model):
    """-> sad [L, 400, D] float32 (layer-major), framing fp [1080, L, D], names, tasks, y."""
    tmpl = LADDERS[ladder][0]
    if tmpl == "GSTUDY":
        g = ACT / "results" / "gstudy" / model
        with np.load(g / "pooled.npz", allow_pickle=False) as z:
            fr = z["framing"]
            names = np.array([str(s) for s in z["framing_name"]])
            tasks = np.array([str(s) for s in z["framing_task"]])
        with np.load(g / "sad_filtered_lf.npz", allow_pickle=False) as z:
            sad, y = z["sad"], z["sad_label"].astype(int)
    else:
        with np.load(ACT / tmpl.format(model), allow_pickle=False) as z:
            fr = z["framing_validated"]
            sad, y = z["sad_validated"], z["sad_label"].astype(int)
            names = np.array([str(s) for s in z["framing_name"]])
            tasks = np.array([str(s) for s in z["framing_task"]])
    sad = np.ascontiguousarray(np.asarray(sad, dtype=np.float32).transpose(1, 0, 2))
    return sad, fr, names, tasks, y


def wrapper_rows(names, tasks):
    ev = sorted({n for n in set(names) if n.startswith("eval_")})
    dp = sorted({n for n in set(names) if n.startswith("deploy_")})
    task_order = sorted(set(tasks[names == ev[0]]))
    rows = {}
    for f in ev + dp:
        idx = np.flatnonzero(names == f)
        by_task = {t: i for i, t in zip(idx, tasks[idx])}
        assert sorted(by_task) == task_order, f"framing {f} does not cover the same tasks"
        rows[f] = np.array([by_task[t] for t in task_order])
    pairs = [(e, d) for e in ev for d in dp]
    return pairs, ev, dp, rows, len(task_order)


def weights(pairs, rows, n_rows, n_tasks):
    """Real wrapper weights and null weights over the framing rows, identical for every model."""
    real = np.zeros((n_rows, len(pairs)), np.float32)
    for j, (fe, fd) in enumerate(pairs):
        real[rows[fe], j] = 1.0 / n_tasks
        real[rows[fd], j] = -1.0 / n_tasks
    rng = np.random.default_rng(SEED_NULL)
    nulls = {k: np.zeros((n_rows, N_NULL * len(pairs)), np.float32) for k in ("signflip", "permute", "cov")}
    for b in range(N_NULL):
        for j, (fe, fd) in enumerate(pairs):
            c = b * len(pairs) + j
            s = rng.choice([-1.0, 1.0], size=n_tasks)
            nulls["signflip"][rows[fe], c] = s / n_tasks
            nulls["signflip"][rows[fd], c] = -s / n_tasks
            both = np.concatenate([rows[fe], rows[fd]])
            perm = rng.permutation(both.size)
            nulls["permute"][both[perm[:n_tasks]], c] = 1.0 / n_tasks
            nulls["permute"][both[perm[n_tasks:]], c] = -1.0 / n_tasks
            z = rng.standard_normal(n_rows)
            nulls["cov"][:, c] = (z - z.mean()) / np.sqrt(n_rows)
    return real, nulls


def score_cells(y, proj, blocks, halves, n_halves):
    """proj [400, L, C] -> dict of cell arrays: full [C], blocks [C, nb], halves [n_halves, 2, C]."""
    out = {"full": peaks(y, proj)}
    out["blocks4"] = np.stack([peaks(y[b], proj[b]) for b in blocks[4]], axis=1)
    out["halves"] = np.stack([np.stack([peaks(y[h], proj[h]) for h in pair])
                              for pair in halves[:n_halves]])
    return out


def run_ladder(ladder):
    tmpl, params = LADDERS[ladder]
    models = list(params)
    if tmpl != "GSTUDY":
        missing = [m for m in models if not (ACT / tmpl.format(m)).exists()]
        if missing:
            print(f"  SKIP {ladder}: {missing[0]} not under {ACT}")
            return None
    res = {}
    meta = None
    for i, m in enumerate(models):
        t0 = time.time()
        sad, fr, names, tasks, y = load_model(ladder, m)
        pairs, ev, dp, rows, n_tasks = wrapper_rows(names, tasks)
        if meta is None:
            blocks, halves = item_partitions(y)
            boot = bootstrap_indices(y)
            real_w, null_w = weights(pairs, rows, names.size, n_tasks)
            meta = dict(y=y, pairs=pairs, ev=ev, dp=dp, names=names, tasks=tasks)
        else:
            assert np.array_equal(y, meta["y"]) and pairs == meta["pairs"]
            assert np.array_equal(names, meta["names"]) and np.array_equal(tasks, meta["tasks"])
        L, n, D = sad.shape
        G = np.empty((L, n, names.size), np.float32)
        rng_iso = np.random.default_rng([SEED_NULL, i])
        n_iso = N_NULL * len(pairs)
        iso = np.empty((L, n, n_iso), np.float32)
        iso_rep = np.empty((L, n, n_iso), np.float32)
        R_rep = np.random.default_rng([SEED_NULL, i, 1]).standard_normal((D, n_iso),
                                                                          dtype=np.float32)
        diag_layers = set(range(0, L, max(1, L // 6)))
        cosines = {k: [] for k in NULLS}
        for li in range(L):
            F = np.asarray(fr[:, li, :], dtype=np.float32)
            G[li] = sad[li] @ F.T
            R = rng_iso.standard_normal((D, n_iso), dtype=np.float32)
            iso[li] = sad[li] @ R
            iso_rep[li] = sad[li] @ R_rep
            if li in diag_layers:
                # how much of the real wrapper direction each null still contains: |cos| between a
                # null draw and the real direction of the wrapper it stands in for
                dr = F.T @ real_w                                         # [D, 36]
                dr_t = np.tile(dr, (1, N_NULL))
                nr = np.linalg.norm(dr_t, axis=0)
                for k, W in (("signflip", null_w["signflip"]), ("permute", null_w["permute"]),
                             ("cov", null_w["cov"]), ("iso", R), ("iso_rep", R_rep)):
                    dn = W if k.startswith("iso") else F.T @ W
                    cs = np.abs((dn * dr_t).sum(0)) / (np.linalg.norm(dn, axis=0) * nr)
                    cosines[k].append(float(np.median(cs)))
        del sad, fr

        def proj_of(W):
            return np.ascontiguousarray((G @ W).transpose(1, 0, 2))      # [400, L, C]

        r = {}
        P = proj_of(real_w)
        real = score_cells(y, P, blocks, halves, N_SPLITS)
        r["real_full"], r["real_blocks4"], r["real_halves"] = (real["full"], real["blocks4"],
                                                              real["halves"])
        for nb in (2, 8):
            r[f"real_blocks{nb}"] = np.stack([peaks(y[b], P[b]) for b in blocks[nb]], axis=1)
        r["boot_full"] = np.stack([peaks(y[f], P[f]) for f, _ in boot])
        r["boot_blocks4"] = np.stack([np.stack([peaks(y[b], P[b]) for b in bl], axis=1)
                                      for _, bl in boot])
        for k in NULLS:
            if k in ("iso", "iso_rep"):
                Pn = np.ascontiguousarray((iso if k == "iso" else iso_rep).transpose(1, 0, 2))
            else:
                Pn = proj_of(null_w[k])
            c = score_cells(y, Pn, blocks, halves, N_SPLITS_NULL)
            C = len(pairs)
            r[f"{k}_full"] = c["full"].reshape(N_NULL, C)
            r[f"{k}_blocks4"] = c["blocks4"].reshape(N_NULL, C, N_BLOCKS)
            r[f"{k}_halves"] = c["halves"].reshape(N_SPLITS_NULL, 2, N_NULL, C)
            r[f"cos_{k}"] = np.array(np.median(cosines[k]))
        del G, iso, iso_rep
        res[m] = r
        print(f"    {ladder} {m}: {L} layers, d={D}  ({time.time() - t0:.0f}s)", flush=True)

    # stack models on axis 0 of every array
    out = {k: np.stack([res[m][k] for m in models]) for k in res[models[0]]}
    out["models"] = np.array(models)
    out["params_b"] = np.array([params[m] for m in models])
    out["pairs"] = np.array(meta["pairs"])
    out["eval_arms"], out["deploy_arms"] = np.array(meta["ev"]), np.array(meta["dp"])
    out["y"] = meta["y"]
    out["boot_items"] = np.stack([f for f, _ in boot])
    out["config"] = np.array(json.dumps({
        "n_boot": N_BOOT, "n_null": N_NULL, "n_splits": N_SPLITS, "n_splits_null": N_SPLITS_NULL,
        "nulls": list(NULLS), "seed_boot": SEED_BOOT, "seed_null": SEED_NULL,
        "statistic": band_label()["statistic"], "layer_band": band_label()["layer_band"],
        "activation_root_relative_paths": tmpl}))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ladder}.npz", **out)
    print(f"  wrote {OUT.relative_to(ROOT).as_posix()}/{ladder}.npz", flush=True)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    only = sys.argv[1:]
    for ladder in LADDERS:
        if only and ladder not in only:
            continue
        run_ladder(ladder)


if __name__ == "__main__":
    main()
