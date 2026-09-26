"""Test the DC/AC decomposition claim (see docs/theory.md).

Theorem 1: Dirichlet-energy spectral metrics are invariant to a token-uniform (DC)
shift and read only the token-relational (AC) component. Prediction P1-P3:

  P1  DC probe  (mean-pooled activation, `mean_norm`)      -> confounded, S ~ 0
  P2  AC probe  (token variance + participation ratio,     -> selective,  S > 0
                 both provably DC-invariant)
  P3  spectral  (smoothness + hfer, Dirichlet-energy)      -> selective,  S > 0, ~ AC

`S = AUC(eval-vs-deploy) - AUC(deploy-vs-placebo)` is the confound-robust selectivity.
The crossover S(DC) < S(AC) ~ S(spectral) pins the effect to the DC/AC split, not to
"spectral magic". Optionally also reports the DC vs AC magnitude of the actual shift
if per-token task activations were saved (`token_acts.pkl`).

Run:  python -m eval_awareness_spectral.decomposition <results_dir>/<model>
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_long, require_cell_metadata
from .baselines import MOMENT_NAMES  # ["mean_norm","token_var","participation_ratio"]
from .separability import TRAJECTORY_STATS, _trajectory, probe_matrix

DC_MOMENTS = ["mean_norm"]                          # token-uniform  -> DC
AC_MOMENTS = ["token_var", "participation_ratio"]   # shift-invariant -> AC
SPECTRAL = ["smoothness_index", "hfer"]             # Dirichlet-energy -> AC (Theorem 1)


def _moment_traj(npz, which) -> pd.DataFrame:
    mom = npz["moments"]  # [N, L, 3]
    cols = [MOMENT_NAMES.index(w) for w in which]
    rows = []
    for i in range(mom.shape[0]):
        row = {"model": str(npz["model"]), "task_id": npz["task_id"][i],
               "condition": npz["condition"][i]}
        for w, j in zip(which, cols):
            for st, val in zip(TRAJECTORY_STATS, _trajectory(mom[i, :, j].astype(float))):
                row[f"{w}_{st}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _spectral_traj(df, metrics, scope="task_subgraph", norm="sym") -> pd.DataFrame:
    cell = df[(df["graph_scope"] == scope) & (df["normalization"] == norm)]
    agg = cell.groupby(["model", "task_id", "condition", "layer"])[metrics].mean().reset_index()
    rows = []
    for (m, t, c), g in agg.groupby(["model", "task_id", "condition"]):
        g = g.sort_values("layer")
        row = {"model": m, "task_id": t, "condition": c}
        for metric in metrics:
            for st, val in zip(TRAJECTORY_STATS, _trajectory(g[metric].to_numpy(float))):
                row[f"{metric}_{st}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _S(feats, n_perm) -> dict:
    cols = [c for c in feats.columns if c not in ("model", "task_id", "condition")]
    ev = probe_matrix(feats, cols, "eval", "deploy", n_boot=1, n_perm=n_perm)
    neg = probe_matrix(feats, cols, "deploy", "placebo", n_boot=1, n_perm=n_perm)
    a = float(ev.auc.iloc[0]) if not ev.empty else np.nan
    n = float(neg.auc.iloc[0]) if not neg.empty else np.nan
    p = float(ev.perm_p.iloc[0]) if not ev.empty else np.nan
    return {"n_features": len(cols), "eval_v_deploy": a, "neg_ctrl": n,
            "selectivity_S": a - n, "eval_perm_p": p}


def shift_magnitude(token_acts_pkl) -> pd.DataFrame:
    """DC vs AC magnitude of the paired shift Δ = X_a − X_b, per contrast.

    token_acts_pkl: list of {task_id, condition, tok[L+1, n, d]} (task-subgraph tokens).
    Prediction: the negative-control shift (deploy-vs-placebo) is MORE DC-dominated
    (higher DC fraction) than the eval-vs-deploy shift.
    """
    recs = pickle.load(open(token_acts_pkl, "rb"))
    by = {}
    for r in recs:
        by.setdefault(r["task_id"], {})[r["condition"]] = np.asarray(r["tok"], float)
    out = []
    for a, b, role in [("eval", "deploy", "eval_signal"), ("deploy", "placebo", "neg_control")]:
        dc, ac = [], []
        for t, cs in by.items():
            if a not in cs or b not in cs or cs[a].shape != cs[b].shape:
                continue
            D = cs[a] - cs[b]                       # [L+1, n, d]
            dc_comp = D.mean(axis=1, keepdims=True)  # token-uniform per layer
            ac_comp = D - dc_comp
            dc.append(np.linalg.norm(dc_comp))
            ac.append(np.linalg.norm(ac_comp))
        if dc:
            dc, ac = np.array(dc), np.array(ac)
            out.append({"contrast": f"{a}_vs_{b}", "role": role, "n_tasks": len(dc),
                        "dc_mag": float(dc.mean()), "ac_mag": float(ac.mean()),
                        "dc_fraction": float((dc / (dc + ac + 1e-12)).mean())})
    return pd.DataFrame(out)


def run(results_model_dir, n_perm=1000):
    d = Path(results_model_dir)
    df = load_long(d / "diagnostics_long.csv")
    npz_path = d / "activations.npz"
    if df.empty or not npz_path.exists():
        print("missing diagnostics or activations at", d); return {}
    require_cell_metadata(df)
    nz = np.load(npz_path, allow_pickle=True)
    npz = {k: nz[k] for k in nz.files}; npz["model"] = str(npz["model"])

    rows = []
    rows.append({"probe": "DC (mean_norm)", **_S(_moment_traj(npz, DC_MOMENTS), n_perm)})
    rows.append({"probe": "AC (token_var+PR)", **_S(_moment_traj(npz, AC_MOMENTS), n_perm)})
    rows.append({"probe": "spectral (smooth+hfer)", **_S(_spectral_traj(df, SPECTRAL), n_perm)})
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "decomposition_crossover.csv", index=False)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pd.set_option("display.width", 200)
    print("\n===== DC/AC PROBE CROSSOVER (Theorem 1 prediction P1-P3) =====")
    print("  want: DC selectivity ~ 0 (confounded); AC & spectral selectivity > 0")
    print(tab.round(3).to_string(index=False))

    tok = d / "token_acts.pkl"
    if tok.exists():
        sm = shift_magnitude(tok)
        sm.to_csv(d / "decomposition_shift_magnitude.csv", index=False)
        print("\n===== SHIFT DC/AC MAGNITUDE (want neg-control MORE DC-dominated) =====")
        print(sm.round(3).to_string(index=False))
    return {"crossover": tab}


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".",
        n_perm=int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
