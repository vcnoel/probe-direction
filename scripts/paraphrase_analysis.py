"""Paraphrased framings: P7a to P7d of docs/paraphrase-registration.md, with its amendment 1.

Per ladder (Qwen3 and Qwen3.5, non-thinking rendering) and per model whose paraphrase extraction
exists (scripts/extract_paraphrase.py):

  gate  the 12 original framings of the new extraction against the paper's stored arrays: max
        absolute activation difference, and the paper's statistic of the 36 version-0 wrappers
        from both. Registered: any wrapper differing by more than 0.01 stops the analysis.
        Amendment 2 (the registered rule failed on bf16 noise): every version-0 direction within
        cosine 0.995 of the stored one at every layer, mean |d fold| <= 0.005, and max |d fold|
        <= max(0.01, 2 x the same-machine batch floor of scripts/paraphrase_noise_floor.py).
        Both verdicts are recorded per model.
  P7a   Spearman across the 36 wrappers between versions 0-1, 0-2 and 1-2, with a bootstrap over
        evaluation and deployment framings (2000 draws, seed 0).
  P7b   variance components of x[model, wrapper, version] (gstudy_gtheory.components_3way).
  P7c   scaling-sign agreement between versions against 1000 wrapper re-pairings (>= 4 rungs).
  P7d   Amendment 1 orientation of docs/controlled-transfer-registration.md with directions from
        the original framings on train tasks and test framings in version v on held-out tasks,
        against the crossing-partition baseline (controlled_transfer.crossing_partitions).

Directions are full-task (P7a-c) or train-task (P7d) unit mean differences of the validated array.
fold is the paper's statistic (max over all hidden states of the folded |AUROC - 0.5|) and carries
the verdicts; sign (mean of AUROC - 0.5 over relative depth 0.25 to 0.75) is reported beside it.

    EAS_ACTIVATION_ROOT=<root> python scripts/paraphrase_analysis.py [--ungated]

--ungated computes every ladder whatever its gate and writes paraphrase_analysis_ungated.json,
labelled descriptive; the gated file is the registered result.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from controlled_transfer import BAND, N_SPLITS, SPLIT_SEED, crossing_partitions  # noqa: E402
from gstudy_gtheory import auc_cols, components_3way, total  # noqa: E402
from pod_ladders import LADDERS, peak, unit  # noqa: E402
from result_io import write_result  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
ACTS = ACT / "results" / "post_submission" / "paraphrase" / "acts"
OUT = ROOT / "results" / "post_submission" / "paraphrase" / "paraphrase_analysis.json"
OUT_UNGATED = OUT.with_name("paraphrase_analysis_ungated.json")
CT_JSON = ROOT / "results" / "post_submission" / "controlled_transfer.json"
CELLS = ROOT / "results" / "post_submission" / "cells"
LADDER_KEYS = ("qwen3_nothink", "qwen3.5_nothink")
VERSIONS = (0, 1, 2)
GATE = 0.01                    # registered: max |d fold| over the 36 version-0 wrappers
COS_MIN, MEAN_TOL = 0.995, 0.005   # amendment 2
FLOOR_JSON = ROOT / "results" / "post_submission" / "paraphrase" / "noise_floor.json"
N_BOOT, BOOT_SEED = 2000, 0
N_REPAIR, REPAIR_SEED = 1000, 0


def vname(f, v):
    return f if v == 0 else f"{f}_p{v}"


def rows_by(names, tasks, keys, task_ids):
    pos = {(n, t): i for i, (n, t) in enumerate(zip(names, tasks))}
    return np.array([[pos[(k, t)] for t in task_ids] for k in keys])


def stats(y, proj, band):
    """fold and sign of one direction's projections [items, states]."""
    a = auc_cols(y, proj)
    return peak(y, proj), float((a[band] - 0.5).mean())


def amended_gate(model, g):
    """Amendment 2: direction cosine, mean and floor-calibrated maximum (noise_floor.json)."""
    floor = None
    if FLOOR_JSON.exists():
        f = json.loads(FLOOR_JSON.read_text(encoding="utf-8"))["models"].get(model)
        floor = f["floor_batch"]["max_abs_diff_fold"] if f else None
    tol = max(GATE, 2 * floor) if floor is not None else GATE
    ok = (g["min_direction_cosine"] >= COS_MIN and g["mean_abs_diff_fold"] <= MEAN_TOL
          and g["max_abs_diff_fold"] <= tol)
    return {"floor_batch_max_abs_diff_fold": floor, "amended_max_tolerance": tol,
            "passes_amended": bool(ok), "passes": bool(ok)}


def model_cells(ladder, model):
    tmpl, _ = LADDERS[ladder]
    with np.load(ACT / tmpl.format(model), allow_pickle=False) as z:
        sad = z["sad_validated"].astype(np.float32)
        y = z["sad_label"].astype(int)
        s_names = [str(s) for s in z["framing_name"]]
        s_tasks = [str(s) for s in z["framing_task"]]
        s_fr = z["framing_validated"]
    with np.load(ACTS / f"{model}__nothink.npz", allow_pickle=False) as z:
        names = [str(s) for s in z["framing_name"]]
        tasks = [str(s) for s in z["framing_task"]]
        fr = z["framing_validated"]
        meta = json.loads(str(z["meta"]))
    orig = {n for n in names if not re.search(r"_p[12]$", n)}
    ev = sorted(n for n in orig if n.startswith("eval_"))
    dp = sorted(n for n in orig if n.startswith("deploy_"))
    frs = ev + dp
    task_ids = sorted(set(tasks))
    L = fr.shape[1]
    depth = np.arange(L) / (L - 1)
    band = (depth >= BAND[0]) & (depth <= BAND[1])
    pairs = list(itertools.product(ev, dp))                       # pod_ladders order

    # ---- gate: version-0 rows and statistic against the stored arrays
    idx_new = rows_by(names, tasks, frs, task_ids)
    idx_old = rows_by(s_names, s_tasks, frs, task_ids)
    a_new = fr[idx_new].astype(np.float32)
    a_old = s_fr[idx_old].astype(np.float32)
    dif = np.abs(a_new - a_old)
    rel = np.linalg.norm(a_new - a_old, axis=-1) / np.linalg.norm(a_old, axis=-1)
    del a_new, a_old

    def full_dirs(arr, rows, key_of):
        mu = {k: arr[rows[i]].astype(np.float32).mean(0) for i, k in enumerate(key_of)}
        return mu

    keys_all = [vname(f, v) for v in VERSIONS for f in frs]
    mu_new = full_dirs(fr, rows_by(names, tasks, keys_all, task_ids), keys_all)
    mu_old = full_dirs(s_fr, idx_old, frs)
    fold = np.zeros((3, len(pairs)))
    sign = np.zeros((3, len(pairs)))
    fold_rel = np.zeros((3, len(pairs)))          # folded max over states 1..L-2 (released set)
    fold_old, sign_old = np.zeros(len(pairs)), np.zeros(len(pairs))
    cos_min = 1.0
    for j, (e, d) in enumerate(pairs):
        for v in VERSIONS:
            D = unit(mu_new[vname(e, v)] - mu_new[vname(d, v)])
            proj = np.einsum("ild,ld->il", sad, D)
            fold[v, j], sign[v, j] = stats(y, proj, band)
            fold_rel[v, j] = peak(y, proj[:, 1:L - 1])
            if v == 0:
                D0 = D
        D = unit(mu_old[e] - mu_old[d])
        fold_old[j], sign_old[j] = stats(y, np.einsum("ild,ld->il", sad, D), band)
        cos_min = min(cos_min, float((D0 * D).sum(-1).min()))
    gate = {"max_abs_diff_activation": float(dif.max()), "mean_abs_diff_activation": float(dif.mean()),
            "max_abs_stored_activation": float(np.abs(s_fr[idx_old]).max()),
            "max_rel_row_norm_diff": float(rel.max()),
            "max_abs_diff_fold": float(np.abs(fold[0] - fold_old).max()),
            "max_abs_diff_sign": float(np.abs(sign[0] - sign_old).max()),
            "mean_abs_diff_fold": float(np.abs(fold[0] - fold_old).mean()),
            "min_direction_cosine": cos_min,
            "spearman_fold": float(spearmanr(fold[0], fold_old)[0])}
    gate["passes_registered"] = gate["max_abs_diff_fold"] <= GATE
    gate.update(amended_gate(model, gate))
    del s_fr, sad

    # ---- P7d: framing means per split and band layer; train from version 0, test per version
    rng = np.random.default_rng(SPLIT_SEED)
    splits = [rng.permutation(len(task_ids)) for _ in range(N_SPLITS)]
    half = len(task_ids) // 2
    pp = list(itertools.combinations(range(len(frs)), 2))
    a_i, b_i = np.array([p[0] for p in pp]), np.array([p[1] for p in pp])
    bl = np.flatnonzero(band)
    G = np.empty((3, N_SPLITS, bl.size, len(frs), len(pp)), np.float32)
    idx_v = {v: rows_by(names, tasks, [vname(f, v) for f in frs], task_ids) for v in VERSIONS}
    for li, l in enumerate(bl):
        F = {v: fr[idx_v[v], l, :].astype(np.float64) for v in VERSIONS}
        for s, perm in enumerate(splits):
            m = F[0][:, perm[:half], :].mean(1)
            dtr = unit(m[a_i] - m[b_i])
            for v in VERSIONS:
                G[v, s, li] = F[v][:, perm[half:], :].mean(1) @ dtr.T
    return {"model": model, "fold": fold, "sign": sign, "fold_released": fold_rel,
            "fold_stored_v0": fold_old, "gate": gate,
            "G": G, "pairs_pp": pp, "n_eval": len(ev), "pairs": pairs, "ev": ev, "dp": dp,
            "n_states": L, "n_band": int(bl.size), "extraction_meta": meta}


def orientation(G, pp, n_eval):
    """controlled_transfer.orient_summary's A1 statistic for one G[s, l, w, p]."""
    n_fr = G.shape[2]
    pid = {p: i for i, p in enumerate(pp)}
    diff = G[:, :, :, None, :] - G[:, :, None, :, :]
    pf = (diff > 0).mean(axis=(0, 1))

    def ori(u, v, u2, v2):
        return pf[u2, v2, pid[(u, v)]] if u < v else pf[v2, u2, pid[(v, u)]]

    ev, dp = range(n_eval), range(n_eval, n_fr)
    real = [(e, d) for e in ev for d in dp]
    r_ori = [ori(e, d, e2, d2) for e, d in real for e2, d2 in real if e2 != e and d2 != d]
    b_ori = []
    for g1, g2 in crossing_partitions(n_eval, n_fr):
        dirs = [(a, b) for a in g1 for b in g2]
        b_ori.append(np.mean([ori(a, b, a2, b2) for a, b in dirs for a2, b2 in dirs
                              if a2 != a and b2 != b]))
    Ro, p95 = float(np.mean(r_ori)), float(np.percentile(b_ori, 95))
    return {"real_orientation": Ro, "baseline_median": float(np.median(b_ori)),
            "baseline_p95": p95, "n_partitions": len(b_ori), "pass": bool(Ro > p95)}


def boot_spearman(xa, xb, n_eval, n_dep):
    rng = np.random.default_rng(BOOT_SEED)
    vals, dropped = [], 0
    for _ in range(N_BOOT):
        E, D = rng.integers(0, n_eval, n_eval), rng.integers(0, n_dep, n_dep)
        idx = np.array([e * n_dep + d for e in E for d in D])
        a, b = xa[idx], xb[idx]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            dropped += 1
            continue
        vals.append(spearmanr(a, b)[0])
    vals = np.array(vals)
    return {"rho": float(spearmanr(xa, xb)[0]), "ci95": [float(np.percentile(vals, 2.5)),
                                                         float(np.percentile(vals, 97.5))],
            "n_draws_used": int(vals.size), "n_draws_dropped": dropped}


def p7a(c):
    out = {}
    for stat in ("fold", "sign"):
        x = c[stat]
        r = {f"{a}v{b}": boot_spearman(x[a], x[b], len(c["ev"]), len(c["dp"]))
             for a, b in ((0, 1), (0, 2), (1, 2))}
        r["holds"] = bool(r["0v1"]["ci95"][0] > 0 and r["0v2"]["ci95"][0] > 0)
        out[stat] = r
    return out


def p7b(X):
    comp = components_3way(X)
    tot = total(comp)
    shares = {k: float(v / tot) for k, v in comp.items() if not k.startswith("_")}
    names = {"s_m": "model", "s_t": "wrapper", "s_h": "version", "s_mt": "model x wrapper",
             "s_mh": "model x version", "s_th": "wrapper x version", "s_mth": "three-way"}
    return {"shares": {names[k]: v for k, v in shares.items()},
            "truncated_at_zero": [names[k] for k in comp["_truncated"]],
            "model_x_wrapper_exceeds_three_way": bool(shares["s_mt"] > shares["s_mth"])}


def p7c(X, lp):
    r = np.array([[np.corrcoef(lp, X[:, j, v])[0, 1] for j in range(X.shape[1])] for v in VERSIONS])
    out = {}
    for v in (1, 2):
        real = float(np.mean(np.sign(r[0]) == np.sign(r[v])))
        rng = np.random.default_rng(REPAIR_SEED)
        null = np.array([np.mean(np.sign(r[0]) == np.sign(r[v][rng.permutation(r.shape[1])]))
                         for _ in range(N_REPAIR)])
        out[f"0v{v}"] = {"same_sign_fraction": real, "repair_mean": float(null.mean()),
                         "repair_p95": float(np.percentile(null, 95)),
                         "percentile": float(100 * ((null < real).mean() + 0.5 * (null == real).mean())),
                         "reproduces_beyond_repairing": bool(real > np.percentile(null, 95))}
    out["r_by_version"] = r.tolist()
    return out


def stored_a1(ladder):
    """A1 of the controlled-transfer analysis on the paper's stored arrays, for reference."""
    if not CT_JSON.exists():
        return {}
    d = json.loads(CT_JSON.read_text(encoding="utf-8"))
    try:
        return {r["model"]: r["real_orientation"]
                for r in d["ladders"][ladder]["amendment_1"]["a1"]["per_model"]}
    except KeyError:
        return {}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ungated = "--ungated" in sys.argv
    out_path = OUT_UNGATED if ungated else OUT
    result, pop = {}, []
    for ladder in LADDER_KEYS:
        params = LADDERS[ladder][1]
        models = [m for m in params if (ACTS / f"{m}__nothink.npz").exists()]
        if not models:
            print(f"{ladder}: no paraphrase extraction, skipped")
            continue
        C = []
        for m in models:
            c = model_cells(ladder, m)
            g = c["gate"]
            print(f"  {ladder}/{m}: gate max|d act| {g['max_abs_diff_activation']:.3g} "
                  f"(max|act| {g['max_abs_stored_activation']:.3g}), max|d fold| "
                  f"{g['max_abs_diff_fold']:.2g}, max|d sign| {g['max_abs_diff_sign']:.2g}", flush=True)
            C.append(c)
        bad = [c["model"] for c in C if not c["gate"]["passes"]]
        pop += models
        gates = {c["model"]: c["gate"] for c in C}
        if bad and not ungated:
            print(f"{ladder}: reproduction gate (amendment 2) failed for {bad}; P7 stops on this "
                  f"ladder", flush=True)
            result[ladder] = {"models": models, "status": "stopped by the reproduction gate",
                              "gate_failed": bad, "gates": gates}
            continue
        # the version-0 statistic from the stored arrays against the paper's October cells
        cells_check = None
        cf = CELLS / f"matched_{ladder}.npz"
        if cf.exists():
            with np.load(cf, allow_pickle=False) as z:
                ref = {str(m): i for i, m in enumerate(z["models"])}
                cells_check = float(max(np.abs(c["fold_stored_v0"] - z["real_full"][ref[c["model"]]]).max()
                                        for c in C))
        per_model, ref_a1 = [], stored_a1(ladder)
        for c in C:
            a = p7a(c)
            d = {f"v{v}": orientation(c["G"][v], c["pairs_pp"], c["n_eval"]) for v in VERSIONS}
            d["holds"] = bool(d["v1"]["pass"] and d["v2"]["pass"])
            d["stored_arrays_a1_real_orientation"] = ref_a1.get(c["model"])
            per_model.append({"model": c["model"], "gate": c["gate"], "p7a": a, "p7d": d,
                              "n_states": c["n_states"], "n_band_states": c["n_band"],
                              "fold": c["fold"].tolist(), "sign": c["sign"].tolist(),
                              "fold_released": c["fold_released"].tolist(),
                              "extraction": {k: c["extraction_meta"].get(k) for k in
                                             ("revision", "template_sha12", "prompt_sha256", "batch",
                                              "seconds_load", "seconds_forward", "gpu", "torch",
                                              "transformers")}})
            fa = a["fold"]
            print(f"  {ladder}/{c['model']}: P7a fold rho 0v1 {fa['0v1']['rho']:+.2f} "
                  f"[{fa['0v1']['ci95'][0]:+.2f},{fa['0v1']['ci95'][1]:+.2f}] 0v2 {fa['0v2']['rho']:+.2f} "
                  f"[{fa['0v2']['ci95'][0]:+.2f},{fa['0v2']['ci95'][1]:+.2f}] 1v2 {fa['1v2']['rho']:+.2f} "
                  f"-> {'holds' if fa['holds'] else 'fails'};  P7d Ro v0 {d['v0']['real_orientation']:.3f} "
                  f"v1 {d['v1']['real_orientation']:.3f} (p95 {d['v1']['baseline_p95']:.3f}) "
                  f"v2 {d['v2']['real_orientation']:.3f} (p95 {d['v2']['baseline_p95']:.3f}) "
                  f"-> {'pass' if d['holds'] else 'fail'}", flush=True)
        lp = np.array([np.log10(params[m]) for m in models])
        Xf = np.stack([c["fold"].T for c in C])                     # [model, wrapper, version]
        Xs = np.stack([c["sign"].T for c in C])
        Xr = np.stack([c["fold_released"].T for c in C])
        need = math.ceil(0.75 * len(models))
        lad = {"models": models, "params_b": [params[m] for m in models],
               "status": ("computed although the reproduction gate failed for " + ", ".join(bad)
                          + "; descriptive, not a registered result") if bad else "gate passed",
               "version0_stored_vs_october_cells_max_abs_diff": cells_check,
               "per_model": per_model,
               "p7a": {"holds_on_ladder": all(r["p7a"]["fold"]["holds"] for r in per_model),
                       "sign_statistic_holds_on_ladder": all(r["p7a"]["sign"]["holds"] for r in per_model)},
               "p7b": {"fold": p7b(Xf), "sign": p7b(Xs)},
               "p7d": {"models_needed": need, "n_pass": sum(r["p7d"]["holds"] for r in per_model),
                       "holds_on_ladder": sum(r["p7d"]["holds"] for r in per_model) >= need}}
        if len(models) >= 4:
            # fold as registered; signed band and the released layer set (states 1..L-2) beside it
            lad["p7c"] = {"fold": p7c(Xf, lp), "sign": p7c(Xs, lp), "fold_released": p7c(Xr, lp)}
        else:
            lad["p7c"] = {"status": f"not run: {len(models)} rungs, registered minimum 4 (pod step)"}
        b = lad["p7b"]["fold"]["shares"]
        print(f"{ladder}: P7a {'holds' if lad['p7a']['holds_on_ladder'] else 'fails'} on the ladder; "
              f"P7b fold shares model {b['model']:.3f} wrapper {b['wrapper']:.3f} MxW "
              f"{b['model x wrapper']:.3f} WxV {b['wrapper x version']:.3f} three-way {b['three-way']:.3f}; "
              f"P7d {lad['p7d']['n_pass']}/{len(models)} -> "
              f"{'holds' if lad['p7d']['holds_on_ladder'] else 'fails'}", flush=True)
        result[ladder] = lad
    if not result:
        raise SystemExit("no ladder had a paraphrase extraction")
    write_result(out_path, {"ladders": result, "registration": "docs/paraphrase-registration.md",
                       "run": ("ungated: every ladder computed whatever its gate; descriptive, "
                               "see the registration's note on the ungated run") if ungated
                              else "gated as registered (amendment 2)",
                       "gate_tolerance_fold_registered": GATE,
                       "gate_amended": {"min_direction_cosine": COS_MIN, "mean_abs_diff_fold": MEAN_TOL,
                                        "max_abs_diff_fold": "max(0.01, 2 x floor_batch)"},
                       "bootstrap_draws": N_BOOT,
                       "repairings": N_REPAIR},
                 kind="measured",
                 statistic="fold = max over all hidden states of |max(AUROC, 1-AUROC) - 0.5| on the "
                           "SAD scoring set (verdicts); sign = mean over relative depth 0.25-0.75 of "
                           "(AUROC - 0.5) (descriptive); P7d = Amendment 1 orientation of the "
                           "controlled-transfer registration on held-out tasks",
                 note="post-submission analysis registered before computation; not part of the "
                      "submitted manuscript",
                 pooling="validated (BOS and whitespace-only tokens dropped) for directions, scoring "
                         "and held-out framing rows",
                 layers="ALL for fold; relative depth 0.25-0.75 for sign and P7d",
                 scoring_array="sad_validated of results/pod_qwen3/*__nothink.npz and "
                               "results/pod_qwen35/*__nothink.npz (P7a-c); held-out framing rows of "
                               "results/post_submission/paraphrase/acts/*__nothink.npz (P7d)",
                 data_root="pod + post_submission/paraphrase", model_population=sorted(pop))
    print(f"wrote {out_path.as_posix()}")


if __name__ == "__main__":
    main()
