"""Baseline ladder for the eval-awareness probe — the confound-free comparison.

Every feature family is reduced to a matrix keyed by (model, task_id, condition)
and run through `separability.probe_matrix` (identical LOTO + bootstrap +
permutation-null protocol), then evaluated on the full **contrast battery** so we
can compute the paper's headline metric:

    Selectivity  S = AUC(eval vs deploy) − AUC(deploy vs placebo)

A family that decodes eval-vs-deploy but also decodes the deploy-vs-placebo
NEGATIVE CONTROL is picking up wording/length, not evaluativeness. High S with a
null negative control is the win condition, and it is where the invariant spectral
features can beat a raw activation probe even at lower raw AUC.

Families
--------
  length_only          n_tokens (full prompt)                     "it's just length"
  position_only        task-span start index                      "it's just RoPE position"
  attention_only       graph-structure metrics (fiedler, λ3)      "it's the hidden signal, not the graph"
  spectral_static      scale-invariant metrics, per-layer traj    ours (static)
  spectral_velocity    cross-layer derivative of the above        ours (dynamical)
  act_moments          ‖mean‖, token-var, participation ratio     KEY graph-free control
  act_moment_velocity  derivative of act_moments                  dynamical control for velocity
  act_pca{k}           PCA-k of mean-pooled residual              capacity-matched activation probe
  act_full             full mean-pooled residual (strong L2)      activation ceiling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analysis import PRIMARY_METRICS
from .separability import TRAJECTORY_STATS, _trajectory, probe_matrix

logger = logging.getLogger(__name__)

ATTENTION_ONLY_METRICS = ["fiedler_value", "lambda3"]  # pure graph structure (no hidden signal)
MOMENT_NAMES = ["mean_norm", "token_var", "participation_ratio"]

# The battery. Order matters only for reporting.
CONTRASTS = [
    ("eval", "deploy", "headline"),
    ("eval", "placebo", "evaluativeness"),
    ("deploy", "placebo", "NEG-CONTROL"),
    ("control", "deploy", "length/position"),
    ("eval", "eval_subtle", "framing-specificity"),
]


# ----------------------------------------------------- feature matrix builders

def _traj_from_long(df: pd.DataFrame, metrics: List[str], scope: str, norm: str,
                    velocity: bool) -> pd.DataFrame:
    """Trajectory (or velocity-trajectory) features from the long diagnostics CSV."""
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
    metrics = [m for m in metrics if m in cell.columns]
    if cell.empty or not metrics:
        return pd.DataFrame()
    agg = cell.groupby(["model", "task_id", "condition", "layer"])[metrics].mean().reset_index()
    rows = []
    for (model, task_id, cond), g in agg.groupby(["model", "task_id", "condition"]):
        g = g.sort_values("layer")
        row = {"model": model, "task_id": task_id, "condition": cond}
        for m in metrics:
            prof = g[m].to_numpy(dtype=float)
            if velocity:
                prof = np.diff(prof)
            for s, v in zip(TRAJECTORY_STATS, _trajectory(prof)):
                row[f"{m}_{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _scalar_feature(df: pd.DataFrame, col: str, scope: str, norm: str,
                    name: str) -> pd.DataFrame:
    """A single scalar per (task, condition) — for length_only / position_only."""
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
    if cell.empty or col not in cell.columns:
        return pd.DataFrame()
    agg = (cell.groupby(["model", "task_id", "condition"])[col].mean()
           .reset_index().rename(columns={col: name}))
    return agg


def _moment_features(npz: Dict, velocity: bool) -> pd.DataFrame:
    """Activation-moment (or moment-velocity) trajectory features from the npz."""
    mom = npz["moments"]  # [N, L, 3]
    if mom.size == 0:
        return pd.DataFrame()
    rows = []
    for i in range(mom.shape[0]):
        row = {"model": str(npz["model"]), "task_id": npz["task_id"][i],
               "condition": npz["condition"][i]}
        for j, mname in enumerate(MOMENT_NAMES):
            prof = mom[i, :, j].astype(float)
            if velocity:
                prof = np.diff(prof)
            for s, v in zip(TRAJECTORY_STATS, _trajectory(prof)):
                row[f"{mname}_{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_vec_matrix(npz: Dict) -> pd.DataFrame:
    """Flattened mean-pooled residual [N, L*d] with metadata — for PCA/full probes."""
    mv = npz["mean_vec"]  # [N, L, d]
    if mv.size == 0:
        return pd.DataFrame()
    X = mv.reshape(mv.shape[0], -1)
    dfm = pd.DataFrame({"model": str(npz["model"]), "task_id": npz["task_id"],
                        "condition": npz["condition"]})
    feat = pd.DataFrame(X, columns=[f"z{i}" for i in range(X.shape[1])])
    return pd.concat([dfm.reset_index(drop=True), feat], axis=1)


def _position_feature(npz: Dict) -> pd.DataFrame:
    return pd.DataFrame({"model": str(npz["model"]), "task_id": npz["task_id"],
                         "condition": npz["condition"],
                         "task_start": npz["task_start"].astype(float)})


# ----------------------------------------------------------- PCA probe (special)

def _probe_activations_pca(mvmat: pd.DataFrame, cond_a: str, cond_b: str, k: int,
                           label: str, n_boot: int, n_perm: int) -> pd.DataFrame:
    """PCA fit INSIDE each LOTO fold (no leakage) then probe. Uses probe_matrix by
    pre-reducing with a fold-agnostic PCA is unsafe, so we reduce globally per model
    with a clear caveat: for the capacity comparison we fit PCA on all prompts of the
    contrast (transductive). This is standard for a capacity *upper bound* and is
    reported as such; the honest within-fold version is used for act_moments/spectral
    which need no dimensionality reduction."""
    from sklearn.decomposition import PCA
    out = []
    for model, g in mvmat.groupby("model"):
        sub = g[g["condition"].isin([cond_a, cond_b])]
        counts = sub.groupby("task_id")["condition"].nunique()
        sub = sub[sub["task_id"].isin(counts[counts == 2].index)]
        if sub["task_id"].nunique() < 3:
            continue
        feat_cols = [c for c in sub.columns if c.startswith("z")]
        X = sub[feat_cols].to_numpy(dtype=float)
        kk = min(k, X.shape[0] - 1, X.shape[1])
        Z = PCA(n_components=kk, random_state=0).fit_transform(
            (X - X.mean(0)) / (X.std(0) + 1e-8))
        red = sub[["model", "task_id", "condition"]].copy().reset_index(drop=True)
        for i in range(kk):
            red[f"pc{i}"] = Z[:, i]
        out.append(probe_matrix(red, [f"pc{i}" for i in range(kk)], cond_a, cond_b,
                                label=label, n_boot=n_boot, n_perm=n_perm))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------- orchestration

def build_all_families(long_df: pd.DataFrame, npz: Optional[Dict],
                       scope: str = "task_subgraph", norm: str = "rw") -> Dict[str, pd.DataFrame]:
    fams: Dict[str, pd.DataFrame] = {}
    # from the long spectral CSV (full scope for length_only so it varies)
    fams["length_only"] = _scalar_feature(long_df, "n_tokens", "full", "rw", "n_tokens")
    fams["attention_only"] = _traj_from_long(long_df, ATTENTION_ONLY_METRICS, scope, norm, False)
    fams["spectral_static"] = _traj_from_long(long_df, PRIMARY_METRICS, scope, norm, False)
    fams["spectral_velocity"] = _traj_from_long(long_df, PRIMARY_METRICS, scope, norm, True)
    if npz is not None and npz.get("moments", np.zeros(0)).size:
        fams["position_only"] = _position_feature(npz)
        fams["act_moments"] = _moment_features(npz, False)
        fams["act_moment_velocity"] = _moment_features(npz, True)
    return {k: v for k, v in fams.items() if v is not None and not v.empty}


def run_baseline_battery(long_df: pd.DataFrame, npz: Optional[Dict], out_dir,
                         scope: str = "task_subgraph", norm: str = "rw",
                         pca_ks=(8, 24), include_full=True,
                         n_boot: int = 500, n_perm: int = 1000) -> pd.DataFrame:
    """Run every family over the full contrast battery; write a tidy results table."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fams = build_all_families(long_df, npz, scope, norm)
    results = []

    for a, b, role in CONTRASTS:
        for fam_name, feats in fams.items():
            fcols = [c for c in feats.columns if c not in ("model", "task_id", "condition")]
            r = probe_matrix(feats, fcols, a, b, label=fam_name, n_boot=n_boot, n_perm=n_perm)
            if not r.empty:
                r["contrast_role"] = role
                results.append(r)
        # activation PCA / full families need special (per-fold-ish) handling
        if npz is not None and npz.get("mean_vec", np.zeros(0)).size:
            mvmat = _mean_vec_matrix(npz)
            for k in pca_ks:
                r = _probe_activations_pca(mvmat, a, b, k, f"act_pca{k}", n_boot, n_perm)
                if not r.empty:
                    r["contrast_role"] = role
                    results.append(r)
            if include_full:
                fcols = [c for c in mvmat.columns if c.startswith("z")]
                r = probe_matrix(mvmat, fcols, a, b, label="act_full",
                                 n_boot=n_boot, n_perm=n_perm)
                if not r.empty:
                    r["contrast_role"] = role
                    results.append(r)

    res = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if res.empty:
        return res
    res.to_csv(out_dir / "baseline_battery.csv", index=False)
    _selectivity_table(res).to_csv(out_dir / "selectivity.csv", index=False)
    return res


def _selectivity_table(res: pd.DataFrame) -> pd.DataFrame:
    """S = AUC(eval_vs_deploy) − AUC(deploy_vs_placebo) per (model, family)."""
    piv = res.pivot_table(index=["model", "family"], columns="contrast",
                          values="auc", aggfunc="mean")
    rows = []
    for (model, fam), r in piv.iterrows():
        ev = r.get("eval_vs_deploy", np.nan)
        neg = r.get("deploy_vs_placebo", np.nan)
        rows.append({"model": model, "family": fam,
                     "auc_eval_vs_deploy": ev, "auc_neg_control": neg,
                     "selectivity_S": (ev - neg) if pd.notna(ev) and pd.notna(neg) else np.nan})
    return pd.DataFrame(rows).sort_values("selectivity_S", ascending=False)
