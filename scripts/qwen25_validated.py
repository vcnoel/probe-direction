"""The Qwen2.5 row of the ladder table with validated (BOS- and whitespace-dropped) directions.

Companion analysis of docs/paraphrase-registration.md. The submitted Qwen2.5 row builds its
directions from the all-token framing array of results/gstudy/<model>/pooled.npz and scores them on
sad_filtered_lf.npz. scripts/extract_qwen25_validated.py re-extracted the framing rows with both
conventions in one pass. Here:

  1. reproduction: the re-extracted all-token array against the stored one (max absolute
     difference, and every ladder-table quantity computed from both);
  2. the ladder-table quantities of scripts/pod_ladders.py (same code path, same seeds) under the
     validated convention, next to the submitted four-rung values and to all-token values on the
     same rungs.

The 7B rung is re-extracted on the pod. Until it is, four-rung versions are also reported from the
arrays of results/gstudy_v11 (scripts/extract_model.py's August pass, both conventions from one
forward pass), provided each rung's all-token twin reproduces the stored array: all four rungs
from gstudy_v11, and the three re-extracted rungs with the gstudy_v11 7B rung (provisional).

    EAS_ACTIVATION_ROOT=<root> python scripts/qwen25_validated.py
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
from gstudy_gtheory import coefficients, components_3way, total  # noqa: E402
from pod_ladders import LADDERS, N_BLOCKS, N_SPLITS, peak, spans_both, unit  # noqa: E402
from result_io import write_result  # noqa: E402

ACT = Path(os.environ.get("EAS_ACTIVATION_ROOT", str(ROOT)))
GS, V11 = ACT / "results" / "gstudy", ACT / "results" / "gstudy_v11"
NEW = ACT / "results" / "post_submission" / "qwen25_validated" / "acts"
OUT = ROOT / "results" / "post_submission" / "qwen25_validated" / "table1.json"
PARAMS = LADDERS["qwen2.5_paper"][1]
STAT_TOL = 0.01          # registered gate on the per-wrapper statistic


def keyed(arr, names, tasks, order):
    """Rows of arr reordered to `order`, a list of (framing, task)."""
    pos = {(n, t): i for i, (n, t) in enumerate(zip(names, tasks))}
    return arr[np.array([pos[k] for k in order])]


def load_model(model):
    with np.load(GS / model / "pooled.npz", allow_pickle=False) as z:
        names = [str(s) for s in z["framing_name"]]
        tasks = [str(s) for s in z["framing_task"]]
        fam = json.loads(str(z["families"]))
        arrays = {"stored_alltoken": z["framing"].astype(np.float32)}
    order = list(zip(names, tasks))
    with np.load(GS / model / "sad_filtered_lf.npz", allow_pickle=False) as z:
        sad, y = z["sad"].astype(np.float32), z["sad_label"].astype(int)
    for tag, path in (("new", NEW / f"{model}.npz"), ("v11", V11 / model / "pooled.npz")):
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            n2 = [str(s) for s in z["framing_name"]]
            t2 = [str(s) for s in z["framing_task"]]
            for conv in ("validated", "alltokens"):
                arrays[f"{tag}_{conv}"] = keyed(z[f"framing_{conv}"].astype(np.float32), n2, t2, order)
    ev = sorted(k for k, v in fam.items() if v == "eval")
    dp = sorted(k for k, v in fam.items() if v == "deploy")
    return np.array(names), arrays, sad, y, ev, dp


def projections(fr, names, sad, pairs):
    """pod_ladders.load_qwen25, for any framing array."""
    return {j: np.einsum("ild,ld->il", sad, unit(fr[names == fe].mean(0) - fr[names == fo].mean(0)))
            for j, (fe, fo) in enumerate(pairs)}


def ladder_quantities(P, Y, pairs, ev, dp, lp):
    """The per-ladder body of pod_ladders.main, verbatim in computation and seeds."""
    n_m = len(P)
    full = np.array([[peak(Y, P[i][j]) for j in range(len(pairs))] for i in range(n_m)])
    r = np.array([np.corrcoef(lp, full[:, j])[0, 1] for j in range(full.shape[1])])
    by_eval, by_dep = {}, {}
    for j, (fe, fo) in enumerate(pairs):
        by_eval.setdefault(fe, []).append(j)
        by_dep.setdefault(fo, []).append(j)
    q_pos = float((r > 0).mean())
    rep = {"r_min": float(r.min()), "r_max": float(r.max()), "n_positive": int((r > 0).sum()),
           "n_negative": int((r < 0).sum()),
           "n_eval_framings_spanning": int(sum(spans_both(r, i) for i in by_eval.values())),
           "n_deploy_framings_spanning": int(sum(spans_both(r, i) for i in by_dep.values())),
           "statistic_median": float(np.median(full)),
           "expected_spanning_from_marginal": float(len(ev) * (1 - q_pos ** len(dp) - (1 - q_pos) ** len(dp)))}
    rng = np.random.default_rng(0)
    pos, neg = rng.permutation(np.flatnonzero(Y == 1)), rng.permutation(np.flatnonzero(Y == 0))
    blocks = [np.concatenate([p, q]) for p, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]
    x = np.array([[[peak(Y[b], P[i][j][b]) for b in blocks] for j in range(len(pairs))]
                  for i in range(n_m)])
    c = components_3way(x)
    tot = total(c)
    kt = next((k for k in range(1, 100001) if coefficients(c, k=k, n=N_BLOCKS)[0] >= 0.80), None)
    gst = {"shares": {k: float(v / tot) for k, v in c.items() if not k.startswith("_")},
           "components": {k: float(v) for k, v in c.items() if not k.startswith("_")},
           "truncated_at_zero": list(c["_truncated"]),
           "erho_single": float(coefficients(c, k=1, n=1)[0]),
           "erho_table": float(coefficients(c, k=1, n=N_BLOCKS)[0]),
           "phi_table": float(coefficients(c, k=1, n=N_BLOCKS)[1]),
           "k_for_080": kt, "k_for_080_unreachable": kt is None,
           "erho_asymptote_in_k": float(coefficients(c, k=10 ** 6, n=N_BLOCKS)[0])}
    sc = []
    for _ in range(N_SPLITS):
        pa, na = rng.permutation(pos), rng.permutation(neg)
        h1 = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
        h2 = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
        sl = {}
        for half, idx in (("a", h1), ("b", h2)):
            st = np.array([[peak(Y[idx], P[i][j][idx]) for j in range(len(pairs))] for i in range(n_m)])
            sl[half] = np.array([np.polyfit(lp, st[:, j], 1)[0] for j in range(len(pairs))])
        sc.append(float(np.corrcoef(sl["a"], sl["b"])[0, 1]))
    sc = np.array(sc)
    rel = {"slope_split_half_mean": float(sc.mean()),
           "spearman_brown": float(2 * sc.mean() / (1 + sc.mean()))}
    table = {"r_range": [rep["r_min"], rep["r_max"]], "r_positive": rep["n_positive"],
             "spanning_eval_deploy": [rep["n_eval_framings_spanning"], rep["n_deploy_framings_spanning"]],
             "model_share": gst["shares"]["s_m"], "model_x_wrapper_share": gst["shares"]["s_mt"],
             "model_x_item_share": gst["shares"]["s_mh"], "erho": gst["erho_table"],
             "k_for_080": kt}
    return {"models": None, "table1": table, "replication": rep, "gtheory": gst,
            "slope_reliability": rel, "full_statistic": full.tolist(), "r": r.tolist()}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    models = list(PARAMS)
    P, repro, Y, pairs, ev_, dp_ = {}, {}, None, None, None, None
    for m in models:
        names, A, sad, y, ev, dp = load_model(m)
        if Y is not None:
            assert np.array_equal(Y, y) and ev == ev_ and dp == dp_
        Y, ev_, dp_ = y, ev, dp
        pairs = list(itertools.product(ev, dp))
        for tag, fr in A.items():
            P.setdefault(tag, {})[m] = projections(fr, names, sad, pairs)
        ref = A["stored_alltoken"]
        stat_ref = np.array([peak(y, P["stored_alltoken"][m][j]) for j in range(len(pairs))])
        rr = {"max_abs_stored": float(np.abs(ref).max())}
        for tag in ("new_alltokens", "v11_alltokens"):
            if tag in A:
                d = np.abs(A[tag] - ref)
                rownorm = np.linalg.norm(A[tag] - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
                st = np.array([peak(y, P[tag][m][j]) for j in range(len(pairs))])
                rr[tag] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                           "max_rel_row_norm_diff": float(rownorm.max()),
                           "max_abs_diff_statistic": float(np.abs(st - stat_ref).max()),
                           "passes_gate": bool(np.abs(st - stat_ref).max() <= STAT_TOL)}
        if "new_validated" in A and "v11_validated" in A:
            rr["new_vs_v11_validated_max_abs_diff"] = float(np.abs(A["new_validated"] - A["v11_validated"]).max())
        repro[m] = rr
        print(f"  {m}: " + "  ".join(f"{k} max|d| {v['max_abs_diff']:.3g} stat {v['max_abs_diff_statistic']:.2g}"
                                      for k, v in rr.items() if isinstance(v, dict)), flush=True)
        del A, sad

    def run(tag_by_model, label):
        ms = list(tag_by_model)
        lp = np.array([np.log10(PARAMS[m]) for m in ms])
        res = ladder_quantities([P[tag_by_model[m]][m] for m in ms], Y, pairs, ev_, dp_, lp)
        res["models"] = ms
        res["arrays"] = tag_by_model
        t = res["table1"]
        print(f"  {label:<44} r {t['r_range'][0]:+.2f}..{t['r_range'][1]:+.2f}  r>0 {t['r_positive']:>2}  "
              f"span {t['spanning_eval_deploy'][0]}/{t['spanning_eval_deploy'][1]}  model "
              f"{100 * t['model_share']:.1f}%  MxW {100 * t['model_x_wrapper_share']:.1f}%  "
              f"Erho {t['erho']:.3f}  k080 {t['k_for_080'] or 'never'}", flush=True)
        return res

    have_new = [m for m in models if m in P.get("new_validated", {})]
    gate_ok = all(repro[m]["new_alltokens"]["passes_gate"] for m in have_new)
    out = {"reproduction": repro, "gate_statistic_tolerance": STAT_TOL, "gate_passed": gate_ok,
           "conditions": {}}
    C = out["conditions"]
    C["submitted_stored_alltoken_4rung"] = run({m: "stored_alltoken" for m in models}, "submitted (stored all-token, 4 rungs)")
    sub = json.loads((ROOT / "results" / "pod_gtheory.json").read_text(encoding="utf-8"))["ladders"]["qwen2.5_paper"]
    reprep = json.loads((ROOT / "results" / "pod_replication.json").read_text(encoding="utf-8"))["ladders"]["qwen2.5_paper"]
    g = C["submitted_stored_alltoken_4rung"]
    out["reimplementation_check"] = {
        "max_abs_diff_shares": max(abs(g["gtheory"]["shares"][k] - sub["shares"][k]) for k in sub["shares"]),
        "erho_single_diff": abs(g["gtheory"]["erho_single"] - sub["erho_single"]),
        "r_min_diff": abs(g["replication"]["r_min"] - reprep["r_min"]),
        "r_max_diff": abs(g["replication"]["r_max"] - reprep["r_max"])}
    # float32 projections; a different numpy build moves the last digits only
    assert max(out["reimplementation_check"].values()) < 1e-6, out["reimplementation_check"]
    if all(m in P.get("v11_validated", {}) and repro[m]["v11_alltokens"]["passes_gate"] for m in models):
        # extract_model.py's August pass: both conventions from one forward pass, its all-token
        # twin checked against the stored arrays above
        C["v11_alltoken_4rung"] = run({m: "v11_alltokens" for m in models}, "gstudy_v11 all-token, 4 rungs")
        C["v11_validated_4rung"] = run({m: "v11_validated" for m in models}, "gstudy_v11 validated, 4 rungs")
    if have_new:
        if not gate_ok:
            raise SystemExit("re-extracted all-token array does not reproduce the stored statistic; "
                             "stopping before the validated analysis (registered gate)")
        C[f"stored_alltoken_{len(have_new)}rung"] = run({m: "stored_alltoken" for m in have_new}, "stored all-token, same rungs")
        C[f"new_alltoken_{len(have_new)}rung"] = run({m: "new_alltokens" for m in have_new}, "re-extracted all-token, same rungs")
        C[f"new_validated_{len(have_new)}rung"] = run({m: "new_validated" for m in have_new}, "re-extracted validated, same rungs")
        rest = [m for m in models if m not in have_new]
        if rest and all(m in P.get("v11_validated", {}) and repro[m]["v11_alltokens"]["passes_gate"] for m in rest):
            tags = {m: ("new_validated" if m in have_new else "v11_validated") for m in models}
            C["validated_4rung_provisional_v11_7b"] = run(tags, "validated, 4 rungs (7B from gstudy_v11)")
    write_result(OUT, out, kind="measured",
                 statistic="folded max over layers, as pod_ladders.py: per-wrapper r with log10 "
                           "parameters, 3-way crossed model x wrapper x item-block components, E rho^2 "
                           "for one wrapper on the full item set (k = 1, n = 4)",
                 note="post-submission analysis (docs/paraphrase-registration.md, companion "
                      "analysis); only the pooling of the direction arrays changes",
                 pooling="directions: validated (BOS and whitespace-only tokens dropped) against "
                         "all-token; scoring: sad_filtered_lf (validated) throughout",
                 layers="ALL",
                 scoring_array="results/gstudy/*/sad_filtered_lf.npz",
                 data_root="gstudy + post_submission/qwen25_validated",
                 model_population=models)
    print(f"wrote {OUT.as_posix()}")


if __name__ == "__main__":
    main()
