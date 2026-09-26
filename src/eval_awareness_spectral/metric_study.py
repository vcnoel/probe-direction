"""What does each spectral metric actually retrieve, and do they complement?

For the eval-awareness question we do not want to lean on one lumped 24-dim vector
(a strong metric can be diluted by a noisy/confounded one). This module decomposes
the signal:

  1. PER-METRIC selectivity — for every metric, in static and velocity (cross-layer
     derivative) form: eval-vs-deploy AUC, the deploy-vs-placebo NEGATIVE-CONTROL
     AUC, and selectivity S = AUC_eval − AUC_neg. S is the confound-robust score:
     it subtracts off whatever wording/filler/position the negative control shares
     with the headline.
  3. PER-LAYER × metric map — where in depth each metric carries eval information.
  2. COMPLEMENTARITY — pairwise combinations and greedy forward selection *by S*
     (not by raw AUC, so we never select a confounded metric), to find the smallest
     metric set with the highest selective signal, and whether combining beats the
     best single metric.

Run:  python -m eval_awareness_spectral.metric_study <results_dir>/<model>
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .analysis import PRIMARY_METRICS, load_long, require_cell_metadata
from .separability import TRAJECTORY_STATS, _trajectory, _loto_scores, _auc, probe_matrix

# A metric can be USED several ways along the depth axis: its static profile, its
# velocity (1st derivative) or its acceleration (2nd derivative). An "atom" is a
# (metric, form) pair; the study asks which metric AND which form carries eval info.
FORMS = ["static", "velocity", "accel"]
ATOMS: List[Tuple[str, str]] = [(m, f) for m in PRIMARY_METRICS for f in FORMS]


def _apply_form(profile: np.ndarray, form: str) -> np.ndarray:
    if form == "velocity":
        return np.diff(profile)
    if form == "accel":
        return np.diff(profile, n=2)
    return profile


def _atom_name(metric: str, form: str) -> str:
    return f"{metric}[{form}]"


def _atom_features(df: pd.DataFrame, specs: List[Tuple[str, str]],
                   scope: str, norm: str) -> pd.DataFrame:
    """Trajectory features for a set of (metric, form) atoms, uniquely named."""
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
    agg = (cell.groupby(["model", "task_id", "condition", "layer"])[PRIMARY_METRICS]
           .mean().reset_index())
    rows = []
    for (model, tid, cond), g in agg.groupby(["model", "task_id", "condition"]):
        g = g.sort_values("layer")
        row = {"model": model, "task_id": tid, "condition": cond}
        for metric, form in specs:
            prof = _apply_form(g[metric].to_numpy(dtype=float), form)
            tag = {"static": "s", "velocity": "v", "accel": "a"}[form]
            for st, val in zip(TRAJECTORY_STATS, _trajectory(prof)):
                row[f"{metric}_{tag}_{st}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _auc_only(feats: pd.DataFrame, cond_a: str, cond_b: str) -> float:
    cols = [c for c in feats.columns if c not in ("model", "task_id", "condition")]
    sub = feats[feats.condition.isin([cond_a, cond_b])].dropna(subset=cols)
    c = sub.groupby("task_id").condition.nunique()
    sub = sub[sub.task_id.isin(c[c == 2].index)]
    if sub.task_id.nunique() < 3:
        return np.nan
    X = sub[cols].to_numpy(float)
    y = (sub.condition.to_numpy() == cond_a).astype(int)
    return _auc(y, _loto_scores(X, y, sub.task_id.to_numpy(), tune=False))


def per_metric_table(df, scope, norm, n_perm=1000) -> pd.DataFrame:
    rows = []
    for metric, form in ATOMS:
        feats = _atom_features(df, [(metric, form)], scope, norm)
        pr = probe_matrix(feats, [c for c in feats.columns if c not in
                                  ("model", "task_id", "condition")],
                          "eval", "deploy", label=_atom_name(metric, form),
                          n_boot=1, n_perm=n_perm)
        ev = float(pr.auc.iloc[0]) if not pr.empty else np.nan
        pp = float(pr.perm_p.iloc[0]) if not pr.empty else np.nan
        neg = _auc_only(feats, "deploy", "placebo")
        evp = _auc_only(feats, "eval", "placebo")
        rows.append({"atom": _atom_name(metric, form), "metric": metric, "form": form,
                     "eval_v_deploy": ev, "eval_v_placebo": evp,
                     "neg_ctrl": neg, "selectivity_S": ev - neg if np.isfinite(ev + neg) else np.nan,
                     "perm_p": pp})
    return pd.DataFrame(rows).sort_values("selectivity_S", ascending=False)


def per_layer_map(df, scope, norm) -> pd.DataFrame:
    """Single-value (per layer) LOTO AUC for each metric — where eval info lives."""
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
    rows = []
    for metric in PRIMARY_METRICS:
        for layer, g in cell.groupby("layer"):
            piv = (g[g.condition.isin(["eval", "deploy"])]
                   .groupby(["task_id", "condition"])[metric].mean().reset_index())
            c = piv.groupby("task_id").condition.nunique()
            piv = piv[piv.task_id.isin(c[c == 2].index)].dropna(subset=[metric])
            if piv.task_id.nunique() < 3:
                continue
            X = piv[[metric]].to_numpy(float)
            y = (piv.condition.to_numpy() == "eval").astype(int)
            rows.append({"metric": metric, "layer": int(layer),
                         "auc": _auc(y, _loto_scores(X, y, piv.task_id.to_numpy(), tune=False))})
    return pd.DataFrame(rows)


def combination_study(df, scope, norm, n_perm=1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise combos + greedy forward selection by selectivity S."""
    # pairwise (static atoms only, to keep it readable)
    static_atoms = [(m, "static") for m in PRIMARY_METRICS]
    pair_rows = []
    for a, b in combinations(static_atoms, 2):
        feats = _atom_features(df, [a, b], scope, norm)
        ev = _auc_only(feats, "eval", "deploy")
        neg = _auc_only(feats, "deploy", "placebo")
        pair_rows.append({"combo": f"{a[0]}+{b[0]}", "eval_v_deploy": ev,
                          "neg_ctrl": neg, "selectivity_S": ev - neg})
    pairs = pd.DataFrame(pair_rows).sort_values("selectivity_S", ascending=False)

    # greedy forward selection over ALL atoms, maximizing S
    remaining = list(ATOMS)
    chosen: List[Tuple[str, bool]] = []
    best_S = -np.inf
    path = []
    while remaining:
        scored = []
        for atom in remaining:
            feats = _atom_features(df, chosen + [atom], scope, norm)
            ev = _auc_only(feats, "eval", "deploy")
            neg = _auc_only(feats, "deploy", "placebo")
            scored.append((atom, ev - neg, ev, neg))
        scored.sort(key=lambda t: (t[1] if np.isfinite(t[1]) else -np.inf), reverse=True)
        atom, S, ev, neg = scored[0]
        if S <= best_S + 1e-9:
            break
        best_S = S
        chosen.append(atom)
        remaining.remove(atom)
        path.append({"added": _atom_name(*atom),
                     "chosen_set": " + ".join(_atom_name(*c) for c in chosen),
                     "eval_v_deploy": ev, "neg_ctrl": neg, "selectivity_S": S})
    return pairs, pd.DataFrame(path)


def run(results_model_dir, scope="task_subgraph", norm="sym", n_perm=1000) -> dict:
    d = Path(results_model_dir)
    df = load_long(d / "diagnostics_long.csv")
    if df.empty:
        print("no diagnostics found at", d); return {}
    require_cell_metadata(df)

    pm = per_metric_table(df, scope, norm, n_perm)
    pl = per_layer_map(df, scope, norm)
    pairs, greedy = combination_study(df, scope, norm, n_perm)

    pm.to_csv(d / "metric_study_per_metric.csv", index=False)
    pl.to_csv(d / "metric_study_per_layer.csv", index=False)
    pairs.to_csv(d / "metric_study_pairs.csv", index=False)
    greedy.to_csv(d / "metric_study_greedy.csv", index=False)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pd.set_option("display.width", 200)
    print(f"\n===== PER-METRIC (task_subgraph/{norm}); S = eval-vs-deploy - neg-control =====")
    print(pm.round(3).to_string(index=False))
    print("\n===== PEAK LAYER per metric (eval-vs-deploy single-value AUC) =====")
    if not pl.empty:
        peak = pl.loc[pl.groupby("metric").auc.idxmax()][["metric", "layer", "auc"]]
        print(peak.round(3).to_string(index=False))
    print("\n===== PAIRWISE combos (static) by selectivity S =====")
    print(pairs.round(3).to_string(index=False))
    print("\n===== GREEDY forward selection by S (smallest set, highest selective signal) =====")
    print(greedy.round(3).to_string(index=False))
    return {"per_metric": pm, "per_layer": pl, "pairs": pairs, "greedy": greedy}


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".",
        n_perm=int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
