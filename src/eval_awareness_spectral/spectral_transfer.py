"""The one thing spectral does that activations CANNOT: cross-architecture transfer.

Spectral trajectory features are fixed-size graph statistics (resampled to a common
relative-depth grid), so a probe trained on one model can be applied to a model with a
DIFFERENT hidden dimension. An activation probe cannot (its weight vector lives in R^d
and d differs across models). So if a spectral eval-vs-deploy probe transfers across
models above chance, that is a capability activations lack by construction.

For each model we build per-prompt features by interpolating each metric's per-layer
profile (task_subgraph/sym) onto a fixed relative-depth grid, then train a logistic probe
on model A (eval vs deploy) and evaluate its AUC on model B. Off-diagonal AUC > 0.5 is
the spectral-unique result. Activation baseline: N/A across models (dim mismatch).

    python -m eval_awareness_spectral.spectral_transfer
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .analysis import load_long, require_cell_metadata

RESULTS = Path(__file__).resolve().parents[2] / "results"
METRICS = ["smoothness_index", "hfer", "spectral_entropy", "gini_sparsity"]
GRID = 16  # relative-depth resample points -> features are model-dimension & depth agnostic

MODELS = [  # (dir, name, hidden_dim) — note the DIFFERENT dims (why activations can't transfer)
    ("decomp_validation/qwen2.5-0.5b-it", "qwen0.5b", 896),
    ("decomp_validation/llama-3.2-1b-it", "llama1b", 2048),
    ("decomp_validation/gemma-2-2b-it",   "gemma2b", 2304),
    ("exp_pm15/llama-3.2-3b-it", "llama3b", 3072),
]


def model_features(df: pd.DataFrame, scope="task_subgraph", norm="sym") -> pd.DataFrame:
    cell = df[(df.graph_scope == scope) & (df.normalization == norm)]
    metrics = [m for m in METRICS if m in cell.columns]
    agg = cell.groupby(["task_id", "condition", "layer"])[metrics].mean().reset_index()
    rows = []
    for (t, c), g in agg.groupby(["task_id", "condition"]):
        g = g.sort_values("layer")
        L = g.layer.to_numpy(float)
        rel = (L - L.min()) / (L.max() - L.min() + 1e-9)
        grid = np.linspace(0, 1, GRID)
        row = {"task_id": t, "condition": c}
        for m in metrics:
            prof = np.interp(grid, rel, g[m].to_numpy(float))
            for i, v in enumerate(prof):
                row[f"{m}_{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _fit(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def run(cond_a="eval", cond_b="deploy", models=None) -> pd.DataFrame:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    feats: Dict[str, pd.DataFrame] = {}
    cols = None
    for d, name, dim in (models or MODELS):
        p = RESULTS / d / "diagnostics_long.csv"
        if not p.exists():
            continue
        df = load_long(p)
        if df.empty:
            continue
        require_cell_metadata(df)
        f = model_features(df)
        f = f[f.condition.isin([cond_a, cond_b])]
        feats[name] = f
        fc = [c for c in f.columns if c not in ("task_id", "condition")]
        cols = fc if cols is None else [c for c in cols if c in fc]

    names = list(feats)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        Xa = feats[a][cols].to_numpy(float)
        ya = (feats[a].condition.to_numpy() == cond_a).astype(int)
        for b in names:
            Xb = feats[b][cols].to_numpy(float)
            yb = (feats[b].condition.to_numpy() == cond_a).astype(int)
            if a == b:
                # within-model: leave-one-task-out
                from sklearn.model_selection import LeaveOneGroupOut
                groups = feats[a].task_id.to_numpy()
                scores = np.full(len(ya), np.nan)
                for tr, te in LeaveOneGroupOut().split(Xa, ya, groups):
                    if len(np.unique(ya[tr])) < 2:
                        continue
                    sc, clf = _fit(Xa[tr], ya[tr])
                    scores[te] = clf.decision_function(sc.transform(Xa[te]))
                mat.loc[a, b] = _auc(ya, scores)
            else:
                sc, clf = _fit(Xa, ya)
                mat.loc[a, b] = _auc(yb, clf.decision_function(sc.transform(Xb)))

    out = RESULTS / "spectral_transfer"
    out.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out / "transfer_auc_matrix.csv")
    off = mat.where(~np.eye(len(names), dtype=bool))
    pd.set_option("display.width", 200)
    print("\n===== CROSS-ARCHITECTURE SPECTRAL PROBE TRANSFER (eval vs deploy AUC) =====")
    print("rows = train model, cols = test model. Different hidden dims:",
          {n: d for _, n, d in MODELS})
    print("Activation probe CANNOT do this (dim mismatch). Off-diagonal > 0.5 = spectral-unique.")
    print(mat.astype(float).round(3).to_string())
    print(f"\nmean OFF-diagonal (cross-model) transfer AUC = {np.nanmean(off.to_numpy(float)):.3f}")
    print(f"mean diagonal (within-model) AUC             = {np.nanmean(np.diag(mat.to_numpy(float))):.3f}")
    return mat


if __name__ == "__main__":
    run()
