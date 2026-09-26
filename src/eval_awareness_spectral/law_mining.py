"""Batch-generate candidate 'laws' about the frame effect and test them all at once.

We do not propose one theorem and stop; we compute a battery of measurable invariants
from the per-token task activations (eval / deploy / placebo), across models, layers and
tasks, and report which are robust. A 'law' = a quantity that is stable/consistent across
models and layers (low variance, consistent sign, or a clean monotone scaling).

Per task, per layer, with task-token signals X_e, X_d, X_p in R^{n x d}:
  Δe = X_e - X_d   (evaluativeness shift)
  Δc = X_d - X_p   (pure-confound shift: both non-evaluative)
  δe = mean_tok(Δe), δc = mean_tok(Δc)   (DC/mean directions in R^d)

Candidate laws (measured per model, late-layer band + full profile):
  C1  collinearity     cos(δe, δc)                 eval vs confound SAME direction?
  C2  full collinear   cos(vec Δe, vec Δc)         incl. token-relational part
  C3  eval-in-conf     (cos(δe,δc))^2              fraction of eval dir in confound dir
  C4  task-indep eval  mean_{i<j} cos(δe_i, δe_j)  is the eval shift a task-independent direction?
  C5  task-indep conf  mean_{i<j} cos(δc_i, δc_j)
  C6  DC frac eval     ||δe||^2 n / ||Δe||^2       token-uniformity of the eval shift
  C7  DC frac conf     ||δc||^2 n / ||Δc||^2
  C8  norm ratio       ||Δe|| / ||Δc||             relative magnitude of the two shifts
  C9  rel size eval    ||Δe|| / ||X_d||            how big is the frame perturbation
  C10 eval rank        participation ratio of {δe_i} over tasks  (1 = single steering dir)
  C11 layer stability  cos(δe[L], δe[L+1])         is the eval direction stable across depth?

Scaling: every C reported vs model params to see if a law emerges with size.

    python -m eval_awareness_spectral.law_mining
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"
MODELS = [  # (dir, name, params_b, family)
    ("decomp_validation/qwen2.5-0.5b-it", "qwen2.5-0.5b", 0.5, "qwen"),
    ("decomp_validation/qwen2.5-3b-it",   "qwen2.5-3b",   3.0, "qwen"),
    ("decomp_validation/qwen2.5-7b-it",   "qwen2.5-7b",   7.0, "qwen"),
    ("decomp_validation/llama-3.2-1b-it", "llama-3.2-1b", 1.0, "llama"),
    ("decomp_validation/llama-3.2-3b-it", "llama-3.2-3b", 3.0, "llama"),
    ("decomp_validation/gemma-2-2b-it",   "gemma-2-2b",   2.0, "gemma"),
]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else np.nan


def _participation_ratio(vecs):  # vecs: [k, d] unit-normalized rows
    U = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vecs])
    s = np.linalg.svd(U, compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-12))


def mine_model(pkl_path: Path) -> pd.DataFrame:
    recs = pickle.load(open(pkl_path, "rb"))
    by = {}
    for r in recs:
        by.setdefault(r["task_id"], {})[r["condition"]] = np.asarray(r["tok"], dtype=np.float32)
    tasks = [t for t, cs in by.items() if {"eval", "deploy", "placebo"} <= set(cs)]
    if not tasks:
        return pd.DataFrame()
    n_layers = by[tasks[0]]["deploy"].shape[0]

    rows = []
    prev_de = {}
    for L in range(n_layers):
        de_list, dc_list, per = [], [], {}
        for t in tasks:
            Xe, Xd, Xp = by[t]["eval"][L], by[t]["deploy"][L], by[t]["placebo"][L]
            if not (Xe.shape == Xd.shape == Xp.shape):
                continue
            De, Dc = Xe - Xd, Xd - Xp
            de, dc = De.mean(0), Dc.mean(0)
            de_list.append(de); dc_list.append(dc)
            per[t] = dict(
                c1=_cos(de, dc),
                c2=_cos(De.ravel(), Dc.ravel()),
                c6=float((np.linalg.norm(de) ** 2 * De.shape[0]) / (np.linalg.norm(De) ** 2 + 1e-12)),
                c7=float((np.linalg.norm(dc) ** 2 * Dc.shape[0]) / (np.linalg.norm(Dc) ** 2 + 1e-12)),
                c8=float(np.linalg.norm(De) / (np.linalg.norm(Dc) + 1e-12)),
                c9=float(np.linalg.norm(De) / (np.linalg.norm(Xd) + 1e-12)),
                de=de,
            )
        if not per:
            continue
        # pairwise task consistency
        des = [per[t]["de"] for t in per]
        dcs = dc_list
        c4 = np.nanmean([_cos(des[i], des[j]) for i in range(len(des)) for j in range(i + 1, len(des))]) if len(des) > 1 else np.nan
        c5 = np.nanmean([_cos(dcs[i], dcs[j]) for i in range(len(dcs)) for j in range(i + 1, len(dcs))]) if len(dcs) > 1 else np.nan
        # STRESS TEST for C1: matched (same-task, shared deploy anchor) vs mismatched
        # (i!=j, no shared anchor). If matched ~ mismatched, C1 is real; if matched is
        # much more negative, C1 is the shared-anchor mechanical artifact.
        c1_matched = np.nanmean([per[t]["c1"] for t in per])
        c1_mismatch = np.nanmean([_cos(des[i], dcs[j]) for i in range(len(des))
                                  for j in range(len(dcs)) if i != j]) if len(des) > 1 else np.nan
        mean_de = np.mean(des, axis=0)
        c11 = _cos(mean_de, prev_de[L - 1]) if (L - 1) in prev_de else np.nan
        prev_de[L] = mean_de
        rows.append({
            "layer": L, "rel_depth": L / (n_layers - 1),
            "C1_matched": c1_matched, "C1_mismatch": c1_mismatch,
            "C1_collin_dir": np.nanmean([per[t]["c1"] for t in per]),
            "C2_collin_full": np.nanmean([per[t]["c2"] for t in per]),
            "C3_eval_in_conf": np.nanmean([per[t]["c1"] ** 2 for t in per]),
            "C4_taskindep_eval": c4, "C5_taskindep_conf": c5,
            "C6_dcfrac_eval": np.nanmean([per[t]["c6"] for t in per]),
            "C7_dcfrac_conf": np.nanmean([per[t]["c7"] for t in per]),
            "C8_norm_ratio": np.nanmean([per[t]["c8"] for t in per]),
            "C9_relsize_eval": np.nanmean([per[t]["c9"] for t in per]),
            "C10_eval_rank": _participation_ratio(des),
            "C11_layer_stab": c11,
        })
    return pd.DataFrame(rows)


def run(models=None) -> pd.DataFrame:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    per_model = []
    profiles = {}
    for d, name, pb, family in (models or MODELS):
        p = RESULTS / d / "token_acts.pkl"
        if not p.exists():
            continue
        prof = mine_model(p)
        if prof.empty:
            continue
        profiles[name] = prof
        band = prof[prof.rel_depth >= 0.5]  # late-layer band (signal lives late)
        summ = band.drop(columns=["layer", "rel_depth"]).mean(numeric_only=True).to_dict()
        summ = {"model": name, "family": family, "params_b": pb,
                **{k: round(v, 3) for k, v in summ.items()}}
        per_model.append(summ)

    tab = pd.DataFrame(per_model)
    out = RESULTS / "law_mining"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "law_summary_by_model.csv", index=False)
    for name, prof in profiles.items():
        prof.to_csv(out / f"law_profile_{name}.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n############ CANDIDATE LAWS: late-layer band (rel_depth>=0.5), by model ############")
    print(tab.to_string(index=False))
    print("\nHow to read: C1/C2~1 => eval-shift IS the confound-shift (explains non-selectivity).")
    print("C4/C5~1 => the frame is a task-independent linear direction. C10~1 => single steering dir.")
    print("Any column monotone in params_b => a scaling law. Compare eval (C4/C6) vs conf (C5/C7).")
    return tab


if __name__ == "__main__":
    run()
