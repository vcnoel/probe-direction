"""Quantify how far apart two framings push the spectral internals.

Core question: for a fixed model and a fixed base task, does swapping the frame
from "deployment" to "evaluation" move the layer-wise spectral metrics? Each base
task is a **paired** observation (same content, different frame).

v2 structure
------------
Every analysis is scoped to a *cell* = (``graph_scope``, ``normalization``). The
primary cell is ``(task_subgraph, rw)``; ``(full, rw)`` is the reference and
``(task_subgraph, none)`` is the mass-leak demonstrator. Metrics are tagged:

* **primary** (scale-invariant / recomputed): ``smoothness_index``,
  ``spectral_entropy``, ``hfer``, ``gini_sparsity``, ``attention_gini`` —
  claims are read from these.
* **descriptive only** (length-sensitive): ``energy``, ``fiedler_value`` —
  reported, never the headline.
* **directed** (asymmetry): ``spectral_radius``, ``max_imaginary``.

Beyond per-layer effect sizes we compute **spectral velocity** (the cross-layer
discrete derivative — "how internals reorganize with depth") and test it
*functionally* with a cluster-based permutation test over the layer axis, rather
than collapsing it to a fragile arg-max index.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY_METRICS = [
    "smoothness_index",
    "spectral_entropy",
    "hfer",
    "gini_sparsity",   # Gini of the library's spectral_masses (a read-out, not a re-derivation)
]
DESCRIPTIVE_METRICS = ["energy", "fiedler_value"]   # length-sensitive; not primary
DIRECTED_METRICS = ["spectral_radius", "max_imaginary"]
DEFAULT_METRICS = PRIMARY_METRICS + DESCRIPTIVE_METRICS + DIRECTED_METRICS


def load_long(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def require_cell_metadata(df: pd.DataFrame) -> None:
    """Reject untagged legacy rows instead of relabeling random-walk metrics."""
    missing = {"graph_scope", "normalization"} - set(df.columns)
    if missing:
        raise ValueError(
            "legacy diagnostics lack explicit graph_scope/normalization metadata; "
            "they may contain invalid random-walk basis metrics and must be recomputed"
        )


def cells_in(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Distinct (graph_scope, normalization) cells present, primary cell first."""
    if df.empty:
        return []
    require_cell_metadata(df)
    have = (
        df[["graph_scope", "normalization"]].drop_duplicates()
        .itertuples(index=False, name=None)
    )
    have = list(have)
    order = {("task_subgraph", "sym"): 0, ("full", "sym"): 1, ("task_subgraph", "none"): 2}
    return sorted(have, key=lambda c: order.get(c, 99))


# --------------------------------------------------------- per-layer effects

def _paired_table(df: pd.DataFrame, cond_a: str, cond_b: str, metric: str) -> pd.DataFrame:
    """Wide table indexed by (model, task_id, layer) with columns a, b, delta.

    Assumes df is already filtered to a single cell. Runs averaged first.
    """
    sub = df[df["condition"].isin([cond_a, cond_b])].copy()
    sub = sub[["model", "task_id", "layer", "condition", metric]].dropna(subset=[metric])
    agg = (
        sub.groupby(["model", "task_id", "layer", "condition"])[metric]
        .mean().reset_index()
    )
    wide = agg.pivot_table(
        index=["model", "task_id", "layer"], columns="condition", values=metric
    ).reset_index()
    if cond_a not in wide.columns or cond_b not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=[cond_a, cond_b])
    wide["delta"] = wide[cond_a] - wide[cond_b]
    return wide


def _cohen_dz(deltas: np.ndarray) -> float:
    sd = np.std(deltas, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.mean(deltas) / sd)


def layer_effects(
    df: pd.DataFrame,
    cond_a: str = "eval",
    cond_b: str = "deploy",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per (model, metric, layer) paired effect for one cell. BH-FDR within the cell."""
    metrics = metrics or DEFAULT_METRICS
    out = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        wide = _paired_table(df, cond_a, cond_b, metric)
        if wide.empty:
            continue
        for (model, layer), g in wide.groupby(["model", "layer"]):
            deltas = g["delta"].to_numpy()
            n = len(deltas)
            if n < 2:
                continue
            t_stat, p_t = stats.ttest_1samp(deltas, 0.0)
            try:
                p_w = stats.wilcoxon(deltas)[1] if np.any(deltas != 0) else 1.0
            except ValueError:
                p_w = np.nan
            out.append({
                "model": model, "metric": metric, "layer": int(layer), "n": n,
                "mean_delta": float(np.mean(deltas)),
                "mean_a": float(g[cond_a].mean()), "mean_b": float(g[cond_b].mean()),
                "dz": _cohen_dz(deltas), "p_t": float(p_t),
                "p_wilcoxon": float(p_w) if p_w is not None else np.nan,
                "is_primary": metric in PRIMARY_METRICS,
            })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res = res.sort_values("p_t").reset_index(drop=True)
    m = len(res)
    res["p_t_fdr"] = (res["p_t"] * m / (res.index + 1)).clip(upper=1.0)
    res["p_t_fdr"] = res["p_t_fdr"][::-1].cummin()[::-1]
    res["sig_fdr05"] = res["p_t_fdr"] < 0.05
    return res.sort_values(["model", "metric", "layer"]).reset_index(drop=True)


def model_summary(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return effects
    rows = []
    for (model, metric), g in effects.groupby(["model", "metric"]):
        peak = g.loc[g["dz"].abs().idxmax()]
        rows.append({
            "model": model, "metric": metric,
            "is_primary": bool(g["is_primary"].iloc[0]),
            "mean_abs_dz": float(g["dz"].abs().mean()),
            "max_abs_dz": float(g["dz"].abs().max()),
            "peak_layer": int(peak["layer"]), "peak_dz": float(peak["dz"]),
            "n_layers": int(g["layer"].nunique()),
            "n_sig_layers": int(g["sig_fdr05"].sum()),
            "frac_sig_layers": float(g["sig_fdr05"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "is_primary", "mean_abs_dz"], ascending=[True, False, False]
    )


def condition_means(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Per (model, condition, layer, metric) mean/sem for one cell."""
    metrics = metrics or DEFAULT_METRICS
    value_vars = [m for m in metrics if m in df.columns]
    long = df.melt(
        id_vars=["model", "condition", "layer", "task_id", "run"],
        value_vars=value_vars, var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    g = (
        long.groupby(["model", "condition", "metric", "layer"])["value"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


# ------------------------------------------------------------- velocity

def compute_velocity(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Cross-layer discrete derivative per (model, task, condition, cell, metric).

    velocity(layer) = metric(layer) - metric(layer-1). Defined for layer >= 1.
    """
    metrics = metrics or (PRIMARY_METRICS + DESCRIPTIVE_METRICS)
    keys = ["model", "task_id", "condition", "run", "graph_scope", "normalization"]
    frames = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        sub = df[keys + ["layer", metric]].dropna(subset=[metric]).sort_values(keys + ["layer"])
        sub["velocity"] = sub.groupby(keys)[metric].diff()
        sub = sub.dropna(subset=["velocity"])
        sub["metric"] = metric
        frames.append(sub[keys + ["layer", "metric", "velocity"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cluster_permutation(deltas: np.ndarray, layers: np.ndarray,
                         n_perm: int = 2000, t_thresh: Optional[float] = None,
                         seed: int = 0) -> dict:
    """Maris-Oostenveld cluster test on paired per-task differences over the layer axis.

    deltas: [n_tasks, n_layers] paired (cond_a - cond_b) velocity differences.
    Null via sign-flip of each task's whole curve (exchangeability of paired sign).
    Returns the largest observed cluster and its permutation p-value.
    """
    n_tasks, n_layers = deltas.shape
    if n_tasks < 2 or n_layers < 1:
        return {"clusters": [], "max_cluster_mass": 0.0, "p_value": np.nan}
    df_t = n_tasks - 1
    if t_thresh is None:
        t_thresh = float(stats.t.ppf(0.975, df_t))

    def layer_t(d):
        mean = d.mean(axis=0)
        sd = d.std(axis=0, ddof=1)
        se = sd / np.sqrt(n_tasks)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, mean / se, 0.0)
        return t

    def clusters_from_t(t):
        mass, out, cur = [], [], None
        for i, ti in enumerate(t):
            if abs(ti) >= t_thresh:
                if cur is None:
                    cur = [i, i, ti]
                else:
                    cur[1] = i; cur[2] += ti
            else:
                if cur is not None:
                    out.append(cur); mass.append(abs(cur[2])); cur = None
        if cur is not None:
            out.append(cur); mass.append(abs(cur[2]))
        return out, (max(mass) if mass else 0.0)

    obs_t = layer_t(deltas)
    obs_clusters, obs_max = clusters_from_t(obs_t)

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_tasks)[:, None]
        _, null_max[p] = clusters_from_t(layer_t(deltas * signs))
    p_value = float((np.sum(null_max >= obs_max) + 1) / (n_perm + 1)) if obs_max > 0 else np.nan

    clusters = [
        {"layer_start": int(layers[c[0]]), "layer_end": int(layers[c[1]]),
         "mass": float(c[2]), "significant": bool(abs(c[2]) >= np.quantile(null_max, 0.95))}
        for c in obs_clusters
    ]
    return {"clusters": clusters, "max_cluster_mass": float(obs_max),
            "p_value": p_value, "t_thresh": float(t_thresh)}


def velocity_cluster_effects(vel: pd.DataFrame, cond_a: str, cond_b: str,
                             n_perm: int = 2000) -> pd.DataFrame:
    """Per (model, cell, metric): cluster-permutation test on eval-deploy velocity."""
    if vel.empty:
        return pd.DataFrame()
    out = []
    grp = ["model", "graph_scope", "normalization", "metric"]
    for (model, scope, norm, metric), g in vel.groupby(grp):
        piv = (
            g[g["condition"].isin([cond_a, cond_b])]
            .groupby(["task_id", "layer", "condition"])["velocity"].mean()
            .unstack("condition")
        )
        if cond_a not in piv.columns or cond_b not in piv.columns:
            continue
        piv = piv.dropna()
        d = (piv[cond_a] - piv[cond_b]).unstack("layer")
        d = d.dropna(axis=0, how="any")
        if d.shape[0] < 2 or d.shape[1] < 1:
            continue
        layers = d.columns.to_numpy()
        res = _cluster_permutation(d.to_numpy(), layers, n_perm=n_perm)
        sig = [c for c in res["clusters"] if c["significant"]]
        out.append({
            "model": model, "graph_scope": scope, "normalization": norm, "metric": metric,
            "n_tasks": int(d.shape[0]), "max_cluster_mass": res["max_cluster_mass"],
            "p_value": res["p_value"], "n_sig_clusters": len(sig),
            "sig_layer_ranges": "; ".join(f"{c['layer_start']}-{c['layer_end']}" for c in sig),
        })
    return pd.DataFrame(out)


def peak_velocity_stability(vel: pd.DataFrame, condition: str,
                            n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap the peak-|velocity| layer over tasks — report stability, not a point.

    Reports the modal peak layer and the fraction of bootstraps landing there, so a
    fragile arg-max is never presented as a settled estimate.
    """
    if vel.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    out = []
    grp = ["model", "graph_scope", "normalization", "metric"]
    for (model, scope, norm, metric), g in vel.groupby(grp):
        gc = g[g["condition"] == condition]
        piv = gc.groupby(["task_id", "layer"])["velocity"].mean().unstack("layer").dropna(axis=0)
        if piv.shape[0] < 2:
            continue
        layers = piv.columns.to_numpy()
        arr = piv.to_numpy()
        n = arr.shape[0]
        peaks = np.empty(n_boot, dtype=int)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            mean_curve = arr[idx].mean(axis=0)
            peaks[b] = layers[int(np.argmax(np.abs(mean_curve)))]
        vals, counts = np.unique(peaks, return_counts=True)
        mode_layer = int(vals[np.argmax(counts)])
        out.append({
            "model": model, "graph_scope": scope, "normalization": norm, "metric": metric,
            "condition": condition, "peak_layer_mode": mode_layer,
            "peak_layer_stability": float(counts.max() / n_boot),
            "peak_layer_ci_lo": int(np.quantile(peaks, 0.025)),
            "peak_layer_ci_hi": int(np.quantile(peaks, 0.975)),
        })
    return pd.DataFrame(out)


# ------------------------------------------------------------- orchestration

def run_analysis(long_csv, out_dir, cond_a: str = "eval", cond_b: str = "deploy",
                 metrics: Optional[List[str]] = None, n_perm: int = 2000) -> dict:
    """Load long CSV; write per-cell effects/summary/means + velocity tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_long(long_csv)
    if df.empty:
        empty = pd.DataFrame()
        return {"effects": empty, "summary": empty, "means": empty,
                "velocity_clusters": empty, "peak_stability": empty}

    require_cell_metadata(df)

    eff_frames, sum_frames, mean_frames = [], [], []
    for scope, norm in cells_in(df):
        cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
        eff = layer_effects(cell, cond_a=cond_a, cond_b=cond_b, metrics=metrics)
        if not eff.empty:
            eff.insert(1, "graph_scope", scope); eff.insert(2, "normalization", norm)
            eff_frames.append(eff)
            s = model_summary(eff)
            s.insert(1, "graph_scope", scope); s.insert(2, "normalization", norm)
            sum_frames.append(s)
        mn = condition_means(cell, metrics=metrics)
        if not mn.empty:
            mn.insert(1, "graph_scope", scope); mn.insert(2, "normalization", norm)
            mean_frames.append(mn)

    effects = pd.concat(eff_frames, ignore_index=True) if eff_frames else pd.DataFrame()
    summary = pd.concat(sum_frames, ignore_index=True) if sum_frames else pd.DataFrame()
    means = pd.concat(mean_frames, ignore_index=True) if mean_frames else pd.DataFrame()

    vel = compute_velocity(df)
    vel_clusters = velocity_cluster_effects(vel, cond_a, cond_b, n_perm=n_perm)
    peak_stab = pd.concat(
        [peak_velocity_stability(vel, cond_a), peak_velocity_stability(vel, cond_b)],
        ignore_index=True,
    ) if not vel.empty else pd.DataFrame()

    tag = f"{cond_a}_vs_{cond_b}"
    effects.to_csv(out_dir / f"layer_effects_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"model_summary_{tag}.csv", index=False)
    means.to_csv(out_dir / "condition_means.csv", index=False)
    vel_clusters.to_csv(out_dir / f"velocity_clusters_{tag}.csv", index=False)
    peak_stab.to_csv(out_dir / "peak_velocity_stability.csv", index=False)

    return {"effects": effects, "summary": summary, "means": means,
            "velocity_clusters": vel_clusters, "peak_stability": peak_stab}
