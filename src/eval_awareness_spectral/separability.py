"""Separability of eval vs deploy from the spectral internals.

Three analyses, all designed to survive an adversarial reviewer:

1. **Leave-task-out linear probe.** Feature = per-layer scale-invariant metrics on
   the ``task_subgraph`` cell. L2-logistic regression, C tuned by *grouped* inner CV,
   **leave-one-task-out** outer CV (whole tasks held out, so a high AUC cannot be
   task memorization). Reported with the number of prompts/features made explicit,
   a **bootstrap CI over tasks**, and a **permutation null on AUC** (labels shuffled
   within task). A cheap per-layer probe (few features) localizes the effect.

2. **Transfer test.** Train on the *subtle* cue (``eval_subtle`` vs ``deploy``),
   test on the *explicit* frame (``eval`` vs ``deploy``), and vice-versa. Above-
   chance transfer is the key evidence that the geometry encodes "being evaluated"
   rather than one frame's surface form.

3. **Davis-Kahan Fiedler rotation.** Per layer, the principal angle between the
   eval and deploy Fiedler vectors on the *identical* task-node subgraph, reported
   as ``1-|cos θ|`` (``|cos|`` kills the eigenvector sign / π-flip ambiguity). Every
   layer is **gated on the eigengap** g = λ₃-λ₂: the Fiedler vector is only
   identified when g is bounded away from 0, so small-gap layers are flagged
   non-interpretable rather than read as signal (this is the actual content of the
   Davis-Kahan bound). A mismatched-pairing permutation gives the null.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROBE_METRICS = ["smoothness_index", "spectral_entropy", "hfer", "gini_sparsity"]

# Trajectory (MMT) statistics of a per-layer metric profile. Collapsing each
# metric's depth profile to these 6 numbers keeps the probe low-dimensional
# (n_metrics × 6 ≈ 24 features) instead of n_layers × n_metrics (~100), which
# matters when there are only dozens of prompts. Mirrors the agent-safety /
# spectral-tool-use baseline convention.
TRAJECTORY_STATS = ["delta", "range", "flux", "max_jump", "slope", "auc"]


def _trajectory(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return np.zeros(len(TRAJECTORY_STATS))
    diffs = np.abs(np.diff(p))
    slope = float(np.polyfit(np.arange(p.size), p, 1)[0])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return np.array([p[-1] - p[0], float(np.ptp(p)), float(diffs.sum()),
                     float(diffs.max()), slope, float(trapz(p))])


# --------------------------------------------------------------- feature build

def build_features(df: pd.DataFrame, scope: str = "task_subgraph",
                   normalization: str = "sym", metrics: Optional[List[str]] = None,
                   mode: str = "trajectory") -> pd.DataFrame:
    """One row per (model, task_id, condition) of probe features (runs averaged).

    mode="trajectory": per-metric MMT stats  → columns ``{metric}_{stat}`` (compact).
    mode="per_layer":  flattened per-layer    → columns ``layer{L}_{metric}``.
    """
    metrics = metrics or PROBE_METRICS
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == normalization)]
    metrics = [m for m in metrics if m in cell.columns]
    if cell.empty or not metrics:
        return pd.DataFrame()
    agg = (
        cell.groupby(["model", "task_id", "condition", "layer"])[metrics]
        .mean().reset_index()
    )
    if mode == "per_layer":
        long = agg.melt(id_vars=["model", "task_id", "condition", "layer"],
                        value_vars=metrics, var_name="metric", value_name="value")
        long["feat"] = "layer" + long["layer"].astype(str) + "_" + long["metric"]
        return long.pivot_table(index=["model", "task_id", "condition"],
                                columns="feat", values="value").reset_index()

    # trajectory mode
    rows = []
    for (model, task_id, cond), g in agg.groupby(["model", "task_id", "condition"]):
        g = g.sort_values("layer")
        row = {"model": model, "task_id": task_id, "condition": cond}
        for m in metrics:
            stats_vec = _trajectory(g[m].to_numpy())
            for s, v in zip(TRAJECTORY_STATS, stats_vec):
                row[f"{m}_{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _loto_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 tune: bool = True) -> np.ndarray:
    """Leave-one-task-out out-of-fold decision scores (grouped inner CV for C)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.model_selection import LeaveOneGroupOut

    logo = LeaveOneGroupOut()
    scores = np.full(len(y), np.nan)
    uniq_groups = np.unique(groups)
    inner_splits = max(2, min(4, len(uniq_groups) - 1))

    for tr, te in logo.split(X, y, groups):
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty="l2", max_iter=2000, C=0.1))
        if tune and len(np.unique(groups[tr])) >= inner_splits and len(np.unique(y[tr])) == 2:
            gkf = GroupKFold(n_splits=inner_splits)
            gs = GridSearchCV(base, {"logisticregression__C": [0.01, 0.1, 1.0]},
                              scoring="roc_auc", cv=gkf, error_score="raise")
            try:
                gs.fit(X[tr], y[tr], groups=groups[tr])
                model = gs.best_estimator_
            except Exception:
                model = base.fit(X[tr], y[tr])
        else:
            model = base.fit(X[tr], y[tr])
        scores[te] = model.decision_function(X[te])
    return scores


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(scores)
    if len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], scores[m]))


def probe_matrix(feats: pd.DataFrame, feature_cols: List[str],
                 cond_a: str, cond_b: str, label: str = "",
                 n_boot: int = 1000, n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Generic leave-task-out probe over ANY feature matrix.

    ``feats`` must have columns [model, task_id, condition, *feature_cols]. Every
    feature family (spectral, activation-moments, activations-PCA, position, …) is
    run through THIS function so the statistical protocol — LOTO, bootstrap CI over
    tasks, permutation null on AUC — is identical and the comparison is fair.
    """
    if feats.empty or not feature_cols:
        return pd.DataFrame()
    out = []
    rng = np.random.default_rng(seed)
    for model, g in feats.groupby("model"):
        sub = g[g["condition"].isin([cond_a, cond_b])].dropna(subset=feature_cols)
        counts = sub.groupby("task_id")["condition"].nunique()
        paired = counts[counts == 2].index
        sub = sub[sub["task_id"].isin(paired)]
        if sub["task_id"].nunique() < 3:
            continue
        X = sub[feature_cols].to_numpy(dtype=float)
        y = (sub["condition"].to_numpy() == cond_a).astype(int)
        groups = sub["task_id"].to_numpy()
        scores = _loto_scores(X, y, groups)
        auc = _auc(y, scores)
        task_ids = np.unique(groups)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(task_ids, size=len(task_ids), replace=True)
            mask = np.concatenate([np.where(groups == t)[0] for t in pick])
            boot[b] = _auc(y[mask], scores[mask])
        ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])
        null = np.empty(n_perm)
        for p in range(n_perm):
            yp = y.copy()
            for t in task_ids:
                if rng.random() < 0.5:
                    m = groups == t
                    yp[m] = 1 - yp[m]
            null[p] = _auc(yp, _loto_scores(X, yp, groups, tune=False))
        p_val = float((np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (n_perm + 1))
        out.append({
            "model": model, "family": label, "contrast": f"{cond_a}_vs_{cond_b}",
            "n_tasks": int(sub["task_id"].nunique()), "n_prompts": int(len(sub)),
            "n_features": len(feature_cols), "auc": auc,
            "auc_ci_lo": float(ci_lo), "auc_ci_hi": float(ci_hi),
            "perm_p": p_val, "perm_null_mean": float(np.nanmean(null)),
        })
    return pd.DataFrame(out)


def probe_separability(df: pd.DataFrame, cond_a: str = "eval", cond_b: str = "deploy",
                       scope: str = "task_subgraph", normalization: str = "sym",
                       n_boot: int = 1000, n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per model: leave-task-out AUC + bootstrap CI + permutation-null p for cond_a vs cond_b."""
    feats = build_features(df, scope, normalization)
    if feats.empty:
        return pd.DataFrame()
    feat_cols = [c for c in feats.columns if c not in ("model", "task_id", "condition")]
    out = []
    rng = np.random.default_rng(seed)

    for model, g in feats.groupby("model"):
        sub = g[g["condition"].isin([cond_a, cond_b])].dropna(subset=feat_cols)
        # keep only tasks present in BOTH conditions (paired)
        counts = sub.groupby("task_id")["condition"].nunique()
        paired = counts[counts == 2].index
        sub = sub[sub["task_id"].isin(paired)]
        if sub["task_id"].nunique() < 3:
            logger.warning("probe: model %s has <3 paired tasks for %s vs %s — skipping",
                           model, cond_a, cond_b)
            continue

        X = sub[feat_cols].to_numpy(dtype=float)
        y = (sub["condition"].to_numpy() == cond_a).astype(int)
        groups = sub["task_id"].to_numpy()

        scores = _loto_scores(X, y, groups)
        auc = _auc(y, scores)

        # bootstrap CI over tasks (resample whole tasks)
        task_ids = np.unique(groups)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(task_ids, size=len(task_ids), replace=True)
            mask = np.concatenate([np.where(groups == t)[0] for t in pick])
            boot[b] = _auc(y[mask], scores[mask])
        ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

        # permutation null: shuffle labels within task, recompute full LOTO AUC
        null = np.empty(n_perm)
        for p in range(n_perm):
            yp = y.copy()
            for t in task_ids:
                m = groups == t
                if rng.random() < 0.5:
                    yp[m] = 1 - yp[m]
            null[p] = _auc(yp, _loto_scores(X, yp, groups, tune=False))
        p_val = float((np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (n_perm + 1))

        out.append({
            "model": model, "contrast": f"{cond_a}_vs_{cond_b}",
            "graph_scope": scope, "normalization": normalization,
            "n_tasks": int(sub["task_id"].nunique()), "n_prompts": int(len(sub)),
            "n_features": len(feat_cols), "auc": auc,
            "auc_ci_lo": float(ci_lo), "auc_ci_hi": float(ci_hi),
            "perm_p": p_val, "perm_null_mean": float(np.nanmean(null)),
        })
    return pd.DataFrame(out)


def per_layer_probe(df: pd.DataFrame, cond_a: str = "eval", cond_b: str = "deploy",
                    scope: str = "task_subgraph", normalization: str = "sym") -> pd.DataFrame:
    """LOTO AUC using only the (few) scale-invariant metrics at each layer."""
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == normalization)]
    metrics = [m for m in PROBE_METRICS if m in cell.columns]
    if cell.empty or not metrics:
        return pd.DataFrame()
    out = []
    for (model, layer), g in cell.groupby(["model", "layer"]):
        piv = (g[g["condition"].isin([cond_a, cond_b])]
               .groupby(["task_id", "condition"])[metrics].mean().reset_index())
        counts = piv.groupby("task_id")["condition"].nunique()
        paired = counts[counts == 2].index
        piv = piv[piv["task_id"].isin(paired)].dropna(subset=metrics)
        if piv["task_id"].nunique() < 3:
            continue
        X = piv[metrics].to_numpy(dtype=float)
        y = (piv["condition"].to_numpy() == cond_a).astype(int)
        groups = piv["task_id"].to_numpy()
        auc = _auc(y, _loto_scores(X, y, groups, tune=False))
        out.append({"model": model, "graph_scope": scope, "normalization": normalization,
                    "layer": int(layer), "auc": auc, "n_tasks": int(piv["task_id"].nunique())})
    return pd.DataFrame(out)


def transfer_test(df: pd.DataFrame, train_pos: str, train_neg: str,
                  test_pos: str, test_neg: str,
                  scope: str = "task_subgraph", normalization: str = "sym") -> pd.DataFrame:
    """Train probe on (train_pos vs train_neg); evaluate AUC on (test_pos vs test_neg).

    Positive class is the 'eval-like' frame in each pair. Above-chance transfer ⇒
    a shared 'being-evaluated' geometry rather than a frame-specific artifact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = build_features(df, scope, normalization)
    if feats.empty:
        return pd.DataFrame()
    feat_cols = [c for c in feats.columns if c not in ("model", "task_id", "condition")]
    out = []
    for model, g in feats.groupby("model"):
        tr = g[g["condition"].isin([train_pos, train_neg])].dropna(subset=feat_cols)
        te = g[g["condition"].isin([test_pos, test_neg])].dropna(subset=feat_cols)
        if tr["condition"].nunique() < 2 or te["condition"].nunique() < 2:
            continue
        Xtr = tr[feat_cols].to_numpy(dtype=float)
        ytr = (tr["condition"].to_numpy() == train_pos).astype(int)
        Xte = te[feat_cols].to_numpy(dtype=float)
        yte = (te["condition"].to_numpy() == test_pos).astype(int)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty="l2", max_iter=2000, C=0.1))
        clf.fit(Xtr, ytr)
        auc = _auc(yte, clf.decision_function(Xte))
        out.append({
            "model": model, "graph_scope": scope, "normalization": normalization,
            "train": f"{train_pos}_vs_{train_neg}", "test": f"{test_pos}_vs_{test_neg}",
            "auc": auc, "n_train": int(len(tr)), "n_test": int(len(te)),
        })
    return pd.DataFrame(out)


# ------------------------------------------------------- Davis-Kahan rotation

def _cos_abs(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return np.nan
    return float(abs(np.dot(u, v)) / (nu * nv))


def davis_kahan_rotation(fiedler_pkl, cond_a: str = "eval", cond_b: str = "deploy",
                         scope: str = "task_subgraph", normalization: str = "sym",
                         gap_quantile: float = 0.25, n_perm: int = 1000,
                         seed: int = 0) -> pd.DataFrame:
    """Per (model, layer): matched Fiedler rotation 1-|cosθ|, gap-gated, with a
    mismatched-pairing permutation null.

    The null pairs cond_a of task i with cond_b of task j≠i: if the matched rotation
    is not distinguishable from mismatched, the per-task subgraph geometry is not
    carrying a frame-specific signal at that layer.
    """
    path = Path(fiedler_pkl)
    if not path.exists():
        return pd.DataFrame()
    with open(path, "rb") as f:
        records = pickle.load(f)
    fr = pd.DataFrame(records)
    fr = fr[(fr["graph_scope"] == scope) & (fr["normalization"] == normalization)]
    if fr.empty:
        return pd.DataFrame()
    # average runs is not meaningful for eigenvectors → take run 0
    fr = fr[fr["run"] == fr["run"].min()]

    rng = np.random.default_rng(seed)
    out = []
    for (model, layer), g in fr.groupby(["model", "layer"]):
        a = g[g["condition"] == cond_a].set_index("task_id")
        b = g[g["condition"] == cond_b].set_index("task_id")
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 3:
            continue
        matched, gaps = [], []
        for t in common:
            ua, ub = a.loc[t, "fiedler_vec"], b.loc[t, "fiedler_vec"]
            if len(ua) != len(ub):
                continue
            matched.append(1.0 - _cos_abs(ua, ub))
            ga = a.loc[t, "lambda3"] - a.loc[t, "lambda2"]
            gb = b.loc[t, "lambda3"] - b.loc[t, "lambda2"]
            gaps.append(np.nanmin([ga, gb]))
        if len(matched) < 3:
            continue
        matched = np.array(matched, float)
        gaps = np.array(gaps, float)
        mean_gap = float(np.nanmean(gaps))

        # mismatched-pairing null
        null_means = np.empty(n_perm)
        avecs = [a.loc[t, "fiedler_vec"] for t in common]
        bvecs = [b.loc[t, "fiedler_vec"] for t in common]
        for p in range(n_perm):
            perm = rng.permutation(len(common))
            vals = [1.0 - _cos_abs(avecs[i], bvecs[perm[i]])
                    for i in range(len(common)) if perm[i] != i
                    and len(avecs[i]) == len(bvecs[perm[i]])]
            null_means[p] = np.mean(vals) if vals else np.nan
        obs = float(np.mean(matched))
        p_val = float((np.nansum(null_means <= obs) + 1) / (n_perm + 1))  # matched SMALLER?

        out.append({
            "model": model, "graph_scope": scope, "normalization": normalization,
            "layer": int(layer), "n_tasks": len(matched),
            "mean_rotation": obs, "rotation_ci_lo": float(np.percentile(matched, 2.5)),
            "rotation_ci_hi": float(np.percentile(matched, 97.5)),
            "mean_eigengap": mean_gap,
            "null_mean_rotation": float(np.nanmean(null_means)),
            "perm_p_matched_lt_null": p_val,
        })
    res = pd.DataFrame(out)
    if not res.empty:
        # gate: a layer is interpretable only if its gap clears the low-gap quantile
        thresh = res["mean_eigengap"].quantile(gap_quantile)
        res["gap_interpretable"] = res["mean_eigengap"] >= thresh
        res["gap_threshold"] = float(thresh)
    return res


def run_separability(long_csv, fiedler_pkl, out_dir,
                     cond_a: str = "eval", cond_b: str = "deploy",
                     scope: str = "task_subgraph", normalization: str = "sym",
                     n_boot: int = 1000, n_perm: int = 1000) -> dict:
    """End-to-end separability suite; writes CSVs, returns the frames."""
    from .analysis import load_long, require_cell_metadata
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_long(long_csv)
    if df.empty:
        return {}
    require_cell_metadata(df)

    probe = probe_separability(df, cond_a, cond_b, scope, normalization, n_boot, n_perm)
    layer_probe = per_layer_probe(df, cond_a, cond_b, scope, normalization)
    transfer = pd.concat([
        transfer_test(df, "eval_subtle", "deploy", "eval", "deploy", scope, normalization),
        transfer_test(df, "eval", "deploy", "eval_subtle", "deploy", scope, normalization),
    ], ignore_index=True)
    dk = davis_kahan_rotation(fiedler_pkl, cond_a, cond_b, scope, normalization, n_perm=n_perm)

    tag = f"{cond_a}_vs_{cond_b}"
    for name, frame in [(f"probe_{tag}", probe), (f"probe_per_layer_{tag}", layer_probe),
                        ("transfer", transfer), (f"davis_kahan_{tag}", dk)]:
        if frame is not None and not frame.empty:
            frame.to_csv(out_dir / f"{name}.csv", index=False)
    return {"probe": probe, "per_layer_probe": layer_probe, "transfer": transfer,
            "davis_kahan": dk}
