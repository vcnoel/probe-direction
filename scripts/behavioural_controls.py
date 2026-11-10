"""Descriptive controls beside the registered behavioural comparison (post hoc, not registered).

The registered comparison (scripts/behavioural_anchor.py) reads rho(B, F) against geometry-matched content-free
rotations. Two further controls, computed after its result and labelled descriptive, ask what else tracks
behaviour across the 19 models:
  size   Spearman rho between log10 parameter count (cells/matched_<ladder>.npz, params_b) and B, and the
         partial Spearman rho of F and B given size (ranks residualised on the ranks of log10 size).
  taskperm  the task-permutation null of the paper's validity audit, which keeps the framings' words and breaks
         only their labels: for each of its 100 draws (cells/matched_<ladder>.npz, taskperm_full), the
         36-wrapper mean per model, and its Spearman rho with B; median, 95th percentile and the share of draws
         at or above the real rho.
B is sp_large accuracy and F the 36-wrapper mean of the folded statistic, both from behavioural_anchor.json.

    python scripts/behavioural_controls.py
Output: results/post_submission/behavioural_controls.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / "results" / "post_submission"
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402

LADDERS = ("qwen3_nothink", "qwen3.5_nothink", "olmo2", "qwen2.5_paper")


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    res = lambda a: a - A @ np.linalg.lstsq(A, a, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    d = json.load(open(PS / "behavioural_anchor.json", encoding="utf-8"))
    models = d["models"]
    F = np.array(d["F"], float)
    B = np.array(d["variants"]["sp_large"]["B"], float)
    params, tp = {}, {}
    for lad in LADDERS:
        with np.load(PS / "cells" / f"matched_{lad}.npz", allow_pickle=False) as z:
            for i, m in enumerate(str(s) for s in z["models"]):
                params[m] = float(z["params_b"][i])
                tp[m] = z["taskperm_full"][i, :100, :].mean(1)
    logp = np.log10([params[m] for m in models])
    rho_fb = float(spearmanr(F, B).correlation)
    T = np.array([tp[m] for m in models]).T
    r_tp = np.array([spearmanr(t, B).correlation for t in T])
    out = {
        "label": "descriptive, post hoc: computed after the registered behavioural comparison",
        "variant": "sp_large", "n_models": len(models), "models": models,
        "rho_F_B": rho_fb,
        "size": {"rho_logparams_B": float(spearmanr(logp, B).correlation),
                 "rho_logparams_F": float(spearmanr(logp, F).correlation),
                 "partial_rho_F_B_given_size": partial_spearman(F, B, logp)},
        "taskperm": {"n_draws": int(len(r_tp)), "median": float(np.median(r_tp)),
                     "p95": float(np.percentile(r_tp, 95)),
                     "share_at_or_above_real": float(np.mean(r_tp >= rho_fb)), "rho_draws": r_tp.tolist()},
    }
    import hashlib
    out["cells_sha256"] = {f"matched_{lad}.npz": hashlib.sha256((PS / "cells" / f"matched_{lad}.npz").read_bytes()).hexdigest()
                           for lad in LADDERS}          # binary inputs: hashed here, JSON sources by write_result
    srcs = ["post_submission/behavioural_anchor.json"]
    write_result(PS / "behavioural_controls.json", out, kind="derived", sources=srcs,
                 statistic="Spearman rho across models: size and the task-permutation null against SAD accuracy",
                 note="descriptive, post hoc controls beside the registered behavioural anchor")
    print(json.dumps({k: v for k, v in out.items() if k not in ("taskperm", "models")}, indent=1))
    print("taskperm median %.3f p95 %.3f share %.2f" % (out["taskperm"]["median"], out["taskperm"]["p95"],
                                                         out["taskperm"]["share_at_or_above_real"]))


if __name__ == "__main__":
    main()
