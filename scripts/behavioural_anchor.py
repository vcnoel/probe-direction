"""Behavioural anchor: the registered comparison (docs/behavioural-anchor-registration.md, Amendment 1).

B(m) is SAD stages_oversight accuracy (sp_large primary, plain beside it with its own
preconditions), read from the sanitised per-model files of scripts/sad_behaviour.py. F, S and O are
the probe statistics of the registration. For each of F, S and O and each variant:

- pooled Spearman rho across the included models;
- one-sided p from 100,000 permutations of B within each ladder (seed 0);
- 95 percent percentile interval from 2,000 bootstraps of the task's 400 items (seed 0), B and the
  probe arrays resampled jointly through results/post_submission/sad_item_index.json; a drawn task
  testing item that a probe array does not hold is replaced, in that array, by a uniform draw from
  its testing items not in the task (Amendment 1);
- for F, rho against the 100 geometry-matched content-free draws (cells/matched_<ladder>.npz,
  matched_full, draw d on every ladder), as a mid-rank percentile and against the 95th percentile;
- the preconditions (split-half reliability, between-model spread, invalid answers) and the
  registered reading.

Secondaries: rho per wrapper for F; the same-prompt read as a table; the comparison on the 278 items
every probe array shares with the task (Amendment 1). Nothing here reads or writes item text.

    python scripts/behavioural_anchor.py
    python scripts/behavioural_anchor.py --test --behaviour-dir <synthetic> --out <path> \\
        --n-perm 2000 --n-boot 20      # pipeline test on synthetic files only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import auc_cols  # noqa: E402
from result_io import write_result  # noqa: E402
from sad_behaviour import SAD_COMMIT  # noqa: E402

PS = ROOT / "results" / "post_submission"
INDEX = PS / "sad_item_index.json"
BEHAVIOUR = PS / "behaviour"
OUT = PS / "behavioural_anchor.json"
ANCHOR_LADDERS = ("qwen3_nothink", "qwen3.5_nothink", "olmo2", "qwen2.5_paper")
VARIANTS = ("sp_large", "plain")            # sp_large is the primary
N_PERM, N_BOOT, N_SPLIT, N_MATCHED, SEED = 100_000, 2_000, 200, 100, 0
MAX_INVALID, MIN_MODELS, MIN_SD, MIN_RELIABILITY, ALPHA = 0.10, 15, 0.05, 0.5, 0.05
BAND = (0.25, 0.75)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rowwise_pearson(A, b):
    """Pearson r of every row of A with the vector b."""
    A = A - A.mean(1, keepdims=True)
    b = b - b.mean()
    return (A @ b) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b))


def fold_signed(P, y):
    """Per wrapper: the paper's folded statistic and the signed band statistic."""
    n, L, J = P.shape
    a = auc_cols(y, P.reshape(n, L * J)).reshape(L, J)
    depth = np.arange(L) / (L - 1)
    band = (depth >= BAND[0]) & (depth <= BAND[1])
    return np.abs(np.maximum(a, 1.0 - a) - 0.5).max(0), (a[band] - 0.5).mean(0)


# ---------------------------------------------------------------- inputs

def load_index():
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    if idx["sad_commit"] != SAD_COMMIT:
        raise SystemExit("item index built from another SAD commit")
    task = idx["task"]
    maps = {}
    for order, rows in idx["orders"].items():
        pos = np.full(len(task), -1)
        for r in rows:
            if r["task_position"] is not None:
                pos[r["task_position"]] = r["position"]
        unshared = np.array([r["position"] for r in rows if r["task_position"] is None], int)
        missing = np.flatnonzero(pos < 0)
        if len(missing) != len(unshared):
            raise SystemExit(f"{order}: {len(missing)} task items unheld, {len(unshared)} spare")
        if any(task[i]["label"] != "testing" for i in missing) or \
                any(rows[j]["label"] != "testing" for j in unshared):
            raise SystemExit(f"{order}: unshared items are expected to be testing items only")
        maps[order] = (pos, unshared)
    return idx, maps


def load_probes(idx):
    signed = json.loads((PS / "signed_statistic.json").read_text(encoding="utf-8"))["ladders"]
    ct = json.loads((PS / "controlled_transfer.json").read_text(encoding="utf-8"))["ladders"]
    models, ladder_of, arrays, F_w, S_w, O, wrappers, hashes = [], {}, {}, {}, {}, {}, None, {}
    for lad in ANCHOR_LADDERS:
        sl = signed[lad]
        if wrappers is None:
            wrappers = sl["wrappers"]
        if sl["wrappers"] != wrappers:
            raise SystemExit(f"{lad}: wrapper order differs")
        orient = {r["model"]: r["real_orientation"] for r in ct[lad]["amendment_1"]["a1"]["per_model"]}
        for k, m in enumerate(sl["models"]):
            f = PS / "controlled" / f"{lad}__{m}.npz"
            with np.load(f, allow_pickle=False) as z:
                P, y = z["sadproj"].astype(np.float64), z["y"].astype(int)
            fw, sw = fold_signed(P, y)
            dev = max(np.abs(fw - np.array(sl["fold"][k])).max(),
                      np.abs(sw - np.array(sl["signed_band"][k])).max())
            if dev > 1e-9:
                raise SystemExit(f"{m}: recomputed statistics differ from signed_statistic.json "
                                 f"by {dev:.2e}")
            want = np.array([r["label"] == "testing"
                             for r in idx["orders"][idx["order_of_ladder"][lad]]], int)
            if not np.array_equal(y, want):
                raise SystemExit(f"{m}: labels disagree with the item index")
            models.append(m)
            ladder_of[m] = lad
            arrays[m] = (P, y)
            F_w[m], S_w[m], O[m] = fw, sw, float(orient[m])
            hashes[f.name] = sha256(f)
    if len(models) != 19:
        raise SystemExit(f"expected 19 models, found {len(models)}")
    return models, ladder_of, arrays, F_w, S_w, O, wrappers, hashes


def load_matched(models, ladder_of, F_w, wrappers):
    """-> [N_MATCHED, models] 36-wrapper means of the matched content-free draws."""
    out = np.full((N_MATCHED, len(models)), np.nan)
    hashes = {}
    for lad in ANCHOR_LADDERS:
        f = PS / "cells" / f"matched_{lad}.npz"
        with np.load(f, allow_pickle=False) as z:
            ms = [str(s) for s in z["models"]]
            pairs = [f"{e} x {d}" for e, d in z["pairs"]]
            if pairs != wrappers:
                raise SystemExit(f"{lad}: matched-null wrapper order differs")
            mf, real = z["matched_full"], z["real_full"]
        for i, m in enumerate(ms):
            if np.abs(real[i] - F_w[m]).max() > 1e-3:
                raise SystemExit(f"{m}: matched-null real statistic differs from the fold")
            out[:, models.index(m)] = mf[i, :N_MATCHED, :].mean(1)
        hashes[f.name] = sha256(f)
    if np.isnan(out).any():
        raise SystemExit("matched null missing for some model")
    return out, hashes


def load_behaviour(bdir: Path, models, idx, test: bool):
    """-> {variant: (score [models, 400], parsed [models, 400])}, present mask, same-prompt table."""
    task = idx["task"]
    pos_of_id = {}
    for i, t in enumerate(task):
        pos_of_id.setdefault(t["sad_id"], []).append(i)
    score = {v: np.full((len(models), len(task)), np.nan) for v in VARIANTS}
    parsed = {v: np.zeros((len(models), len(task)), bool) for v in VARIANTS}
    present, same, files = np.zeros(len(models), bool), {}, {}
    for k, m in enumerate(models):
        f = bdir / f"{m}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["label"] != ("test" if test else "main") or r["meta"]["n_limit"] is not None:
            raise SystemExit(f"{f.name}: not a registered full run (label {r['label']}, "
                             f"n_limit {r['meta']['n_limit']})")
        if r["meta"]["sad_commit"] != SAD_COMMIT:
            raise SystemExit(f"{f.name}: SAD commit {r['meta']['sad_commit']}")
        for v in VARIANTS:
            rows = r["variants"][v]["per_item"]
            if len(rows) != len(task):
                raise SystemExit(f"{f.name} {v}: {len(rows)} items")
            taken = {}
            for row in rows:
                cands = pos_of_id.get(row["sad_id"], [])
                j = taken.get(row["sad_id"], 0)
                if j >= len(cands):
                    raise SystemExit(f"{f.name} {v}: item {row['sad_id']} not in the task index")
                i = cands[j]
                taken[row["sad_id"]] = j + 1
                if row["body_sha256"] != task[i]["body_sha256"] or row["label"] != task[i]["label"]:
                    raise SystemExit(f"{f.name} {v}: item {row['sad_id']} hash or label mismatch")
                score[v][k, i], parsed[v][k, i] = row["score"], row["parsed"]
            if np.isnan(score[v][k]).any():
                raise SystemExit(f"{f.name} {v}: not every task item scored")
        present[k] = True
        sp = r.get("same_prompt")
        if sp:
            same[m] = {"n": sp["n"], "auroc": sp["auroc"], "auroc_folded": sp["auroc_folded"],
                       "item_files": sp.get("item_files")}
        files[f.name] = sha256(f)
    return score, parsed, present, same, files


# ---------------------------------------------------------------- statistics

def within_ladder_perms(groups, n_draws, rng):
    """[n_draws, n] index matrix permuting positions within each group."""
    n = sum(len(g) for g in groups)
    out = np.empty((n_draws, n), int)
    for g in groups:
        g = np.asarray(g)
        perm = rng.permuted(np.tile(np.arange(len(g)), (n_draws, 1)), axis=1)
        out[:, g] = g[perm]
    return out


def reliability(score_inc, labels, rng):
    rs = []
    test_pos, dep_pos = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    for _ in range(N_SPLIT):
        halves = []
        for pos in (test_pos, dep_pos):
            p = rng.permutation(pos)
            halves.append((p[: len(p) // 2], p[len(p) // 2:]))
        a = np.concatenate([halves[0][0], halves[1][0]])
        b = np.concatenate([halves[0][1], halves[1][1]])
        xa, xb = score_inc[:, a].mean(1), score_inc[:, b].mean(1)
        if xa.std() == 0 or xb.std() == 0:
            rs.append(np.nan)
            continue
        r = np.corrcoef(xa, xb)[0, 1]
        rs.append(2 * r / (1 + r))
    rs = np.array(rs)
    return (float(np.nanmedian(rs)) if np.isfinite(rs).any() else float("nan")), rs


def verdict(rho, p, lo, hi, informative, null_p95=None):
    flags = {"rho_positive": bool(rho > 0), "perm_p_below_alpha": bool(p < ALPHA),
             "interval_excludes_0": bool(lo > 0 or hi < 0)}
    if null_p95 is not None:
        flags["rho_above_matched_p95"] = bool(rho > null_p95)
    if not informative:
        return "uninformative", flags
    if rho > 0 and p < ALPHA and lo > 0 and (null_p95 is None or rho > null_p95):
        return "supports", flags
    if rho < 0 and hi < 0:
        return "anti-aligned", flags
    if (rho <= 0 and lo <= 0 <= hi) or (null_p95 is not None and rho <= null_p95):
        return "undercuts", flags
    return "no registered reading applies", flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviour-dir", type=Path, default=BEHAVIOUR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--test", action="store_true", help="synthetic pipeline test; never the "
                                                       "registered output")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if not a.test and (a.n_perm != N_PERM or a.n_boot != N_BOOT):
        ap.error("the registered analysis uses 100,000 permutations and 2,000 bootstraps")
    if a.test and a.out.resolve() == OUT.resolve():
        ap.error("a test never writes the registered output")

    idx, maps = load_index()
    models, ladder_of, arrays, F_w, S_w, O, wrappers, probe_hashes = load_probes(idx)
    nullF, matched_hashes = load_matched(models, ladder_of, F_w, wrappers)
    score, parsed, present, same, bfiles = load_behaviour(a.behaviour_dir, models, idx, a.test)
    labels = np.array([t["label"] == "testing" for t in idx["task"]], int)
    order_of = {m: idx["order_of_ladder"][ladder_of[m]] for m in models}
    F = np.array([F_w[m].mean() for m in models])
    S = np.array([S_w[m].mean() for m in models])
    Ov = np.array([O[m] for m in models])

    # bootstrap draws of the probe statistics, shared by both variants
    rng = np.random.default_rng(SEED)
    draws = np.empty((a.n_boot, len(labels)), int)
    Fb, Sb = np.empty((a.n_boot, len(models))), np.empty((a.n_boot, len(models)))
    for b in range(a.n_boot):
        d = rng.integers(0, len(labels), len(labels))
        draws[b] = d
        pos = {}
        for order, (p, spare) in maps.items():
            q = p[d].copy()
            miss = q < 0
            q[miss] = rng.choice(spare, size=int(miss.sum()), replace=True)
            pos[order] = q
        for k, m in enumerate(models):
            P, y = arrays[m]
            q = pos[order_of[m]]
            fw, sw = fold_signed(P[q], y[q])
            Fb[b, k], Sb[b, k] = fw.mean(), sw.mean()
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{a.n_boot}", flush=True)

    # the 278 items every probe array shares with the task
    common = np.array(idx["shared_task_positions"], int)
    F278, S278 = np.empty(len(models)), np.empty(len(models))
    for k, m in enumerate(models):
        P, y = arrays[m]
        q = maps[order_of[m]][0][common]
        fw, sw = fold_signed(P[q], y[q])
        F278[k], S278[k] = fw.mean(), sw.mean()

    result = {"registration": "docs/behavioural-anchor-registration.md (with Amendment 1)",
              "test_run": bool(a.test), "models": models,
              "ladder_of": ladder_of, "item_order_of_model": order_of,
              "F": F.tolist(), "S": S.tolist(), "O": Ov.tolist(),
              "F_278": F278.tolist(), "S_278": S278.tolist(),
              "n_permutations": a.n_perm, "n_bootstrap": a.n_boot, "seed": SEED,
              "variants": {}, "behaviour_files_sha256": bfiles,
              "array_sha256": {**probe_hashes, **matched_hashes},
              "same_prompt": same}
    for v in VARIANTS:
        sc = score[v]
        inv = np.array([(~parsed[v][k]).mean() if present[k] else np.nan
                        for k in range(len(models))])
        inc = present & (np.nan_to_num(inv, nan=1.0) <= MAX_INVALID)
        excluded = {m: ("missing" if not present[k] else f"invalid {inv[k]:.3f}")
                    for k, m in enumerate(models) if not inc[k]}
        B = np.full(len(models), np.nan)
        B[present] = sc[present].mean(1)
        out_v = {"B": [None if not present[k] else float(B[k]) for k in range(len(models))],
                 "invalid_rate": [None if not present[k] else float(inv[k])
                                  for k in range(len(models))],
                 "included": [m for k, m in enumerate(models) if inc[k]], "excluded": excluded}
        n_inc = int(inc.sum())
        sd = float(np.std(B[inc], ddof=1)) if n_inc > 1 else float("nan")
        rel, rel_all = reliability(sc[inc], labels, np.random.default_rng(SEED)) if n_inc > 2 \
            else (float("nan"), np.array([]))
        pre = {"split_half_reliability": {"value": rel, "threshold": MIN_RELIABILITY,
                                          "holds": bool(rel >= MIN_RELIABILITY)},
               "between_model_sd": {"value": sd, "threshold": MIN_SD, "holds": bool(sd >= MIN_SD)},
               "models_after_invalid_exclusion": {"value": n_inc, "threshold": MIN_MODELS,
                                                  "holds": bool(n_inc >= MIN_MODELS)}}
        informative = all(c["holds"] for c in pre.values())
        out_v["preconditions"] = pre
        out_v["informative"] = informative
        print(f"[{v}] included {n_inc}/19; reliability {rel:.3f}; SD {sd:.3f}; "
              f"{'informative' if informative else 'UNINFORMATIVE'}", flush=True)
        if n_inc < 3:
            result["variants"][v] = out_v
            continue

        ki = np.flatnonzero(inc)
        groups = []
        for lad in ANCHOR_LADDERS:
            g = [j for j, k in enumerate(ki) if ladder_of[models[k]] == lad]
            if g:
                groups.append(g)
        perms = within_ladder_perms(groups, a.n_perm, np.random.default_rng(SEED))
        rB = rankdata(B[ki])
        Bb = sc[ki][:, draws.T].mean(1).T              # [n_boot, n_inc]
        comps = {}
        for name, X, Xb in (("F", F, Fb), ("S", S, Sb), ("O", Ov, None)):
            rho = spearman(B[ki], X[ki])
            null = rowwise_pearson(rB[perms], rankdata(X[ki]))
            p = float((1 + np.sum(null >= rho - 1e-12)) / (1 + a.n_perm))
            boot = np.array([spearman(Bb[b], (Xb[b, ki] if Xb is not None else X[ki]))
                             for b in range(a.n_boot)])
            lo, hi = (float(q) for q in np.nanpercentile(boot, [2.5, 97.5]))
            c = {"rho": rho, "perm_p_one_sided": p, "boot_95": [lo, hi]}
            p95 = None
            if name == "F":
                mrho = np.array([spearman(B[ki], nullF[d, ki]) for d in range(N_MATCHED)])
                p95 = float(np.percentile(mrho, 95))
                c["matched_null"] = {"rho_draws": mrho.tolist(), "p95": p95,
                                     "mid_rank_percentile": float(
                                         (np.sum(mrho < rho) + 0.5 * np.sum(mrho == rho)) / N_MATCHED)}
            c["reading"], c["flags"] = verdict(rho, p, lo, hi, informative, p95)
            comps[name] = c
            print(f"[{v}] {name}: rho {rho:+.3f}, p {p:.4f}, 95% [{lo:+.3f}, {hi:+.3f}]"
                  + (f", matched p95 {p95:+.3f}" if p95 is not None else "")
                  + f" -> {c['reading']}", flush=True)
        out_v["comparison"] = comps
        rw = np.array([spearman(B[ki], np.array([F_w[models[k]][w] for k in ki]))
                       for w in range(len(wrappers))])
        out_v["per_wrapper_F"] = {"rho": rw.tolist(), "share_positive": float(np.mean(rw > 0)),
                                  "min": float(np.nanmin(rw)), "max": float(np.nanmax(rw))}
        B278 = sc[ki][:, common].mean(1)
        rB278 = rankdata(B278)
        sec = {}
        for name, X in (("F", F278), ("S", S278), ("O", Ov)):
            rho = spearman(B278, X[ki])
            null = rowwise_pearson(rB278[perms], rankdata(X[ki]))
            sec[name] = {"rho": rho, "perm_p_one_sided":
                         float((1 + np.sum(null >= rho - 1e-12)) / (1 + a.n_perm))}
        out_v["shared_278_items"] = {"n_items": int(len(common)),
                                     "n_testing": int(labels[common].sum()), "comparison": sec}
        result["variants"][v] = out_v

    srcs = ["post_submission/sad_item_index.json", "post_submission/signed_statistic.json",
            "post_submission/controlled_transfer.json"]
    if not a.test:
        srcs += [f"post_submission/behaviour/{n}" for n in sorted(bfiles)]
    write_result(a.out, result, kind="derived", sources=srcs,
                 statistic="Spearman rho across models between SAD stages_oversight accuracy and "
                           "the probe statistics F (fold), S (signed band) and O (orientation)",
                 note=("SYNTHETIC PIPELINE TEST: no behavioural data" if a.test else
                       "registered behavioural anchor"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
