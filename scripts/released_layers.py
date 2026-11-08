"""The scaling sign of the paper's statistic under five candidate layer sets.

Post-submission descriptive check (docs/post-submission-checks.md), not registered; first computed
by the adversarial audit. The paper's statistic is the folded maximum of |AUROC - 0.5| over ALL
hidden states, embedding output (state 0) and final output included. The released implementations
of the reanalysed studies maximise over hidden_states[1..L-1] of an L-layer model, dropping both
(scripts/contrastive_control.py, score()). The read-layer check found that the maximum often sits
at state 0. This recomputes the per-wrapper scaling r and the sign counts of Table 1 under:

  all            every hidden state, as submitted
  released       states 1 .. n-2 of the n = L+1 hidden states (the released convention)
  drop_last      states 0 .. n-2
  drop_state0    states 1 .. n-1
  band           relative depth 0.25 to 0.75 (index / (n - 1))

for every ladder and rendering, from the stored scoring-set projections of the 36 full-task
directions (results/post_submission/controlled/<ladder>__<model>.npz, key sadproj).

    python scripts/released_layers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import auc_cols  # noqa: E402
from pod_ladders import LADDERS  # noqa: E402
from result_io import write_result  # noqa: E402

CELLS = ROOT / "results" / "post_submission" / "controlled"
OUT = ROOT / "results" / "post_submission" / "released_layers.json"


def layer_sets(n):
    d = np.arange(n) / (n - 1)
    return {"all": np.arange(n), "released": np.arange(1, n - 1), "drop_last": np.arange(0, n - 1),
            "drop_state0": np.arange(1, n), "band": np.flatnonzero((d >= 0.25) & (d <= 0.75))}


def auroc_curves(ladder, model):
    """-> AUROC [n_states, 36] of the full-task directions, wrapper (e, d) index pairs, y."""
    with np.load(CELLS / f"{ladder}__{model}.npz", allow_pickle=False) as z:
        P, y, real = z["sadproj"].astype(np.float64), z["y"].astype(int), z["real"]
    n, L, J = P.shape
    return auc_cols(y, P.reshape(n, L * J)).reshape(L, J), real


def scaling(lp, X, real):
    """X [models, 36] -> Table 1's sign quantities for one statistic."""
    r = np.array([np.corrcoef(lp, X[:, j])[0, 1] for j in range(X.shape[1])])
    by_e = [np.flatnonzero(real[:, 0] == e) for e in np.unique(real[:, 0])]
    by_d = [np.flatnonzero(real[:, 1] == d) for d in np.unique(real[:, 1])]
    return {"r_min": float(r.min()), "r_max": float(r.max()), "n_positive": int((r > 0).sum()),
            "n_negative": int((r < 0).sum()),
            "eval_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_e)),
            "deploy_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_d)),
            "r_of_36_average": float(np.corrcoef(lp, X.mean(1))[0, 1]),
            "mean_statistic": float(X.mean()), "r": r.tolist()}


def main() -> None:
    out, pop = {}, []
    for ladder, (_, params) in LADDERS.items():
        models = list(params)
        lp = np.array([np.log10(params[m]) for m in models])
        stats, real = {}, None
        for m in models:
            a, rl = auroc_curves(ladder, m)
            assert real is None or np.array_equal(real, rl)
            real = rl
            f = np.abs(a - 0.5)
            for tag, lay in layer_sets(a.shape[0]).items():
                stats.setdefault(tag, []).append(f[lay].max(0))
        out[ladder] = {"models": models,
                       "by_layer_set": {t: scaling(lp, np.array(X), real) for t, X in stats.items()}}
        pop += models
        print(f"{ladder:16s} " + " | ".join(
            f"{t} +{s['n_positive']}/-{s['n_negative']} span {s['eval_fixed_spanning']}/"
            f"{s['deploy_fixed_spanning']} r36 {s['r_of_36_average']:+.2f}"
            for t, s in out[ladder]["by_layer_set"].items()), flush=True)
    write_result(OUT, {"ladders": out, "layer_sets": {
        "all": "every hidden state (submitted)", "released": "states 1..n-2 (hidden_states[1..L-1])",
        "drop_last": "states 0..n-2", "drop_state0": "states 1..n-1",
        "band": "relative depth 0.25-0.75, index / (n - 1)"},
        "check": "post-submission descriptive check, not registered; first computed by the "
                 "adversarial audit (docs/post-submission-checks.md)"},
        kind="measured",
        statistic="the paper's folded max of |AUROC - 0.5| over each layer set; per-wrapper Pearson "
                  "r with log10 parameters and Table 1's sign counts",
        note="n = number of hidden states including the embedding output (state 0)",
        pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
        layers="five sets: all, 1..n-2, 0..n-2, 1..n-1, relative depth 0.25-0.75",
        scoring_array="results/post_submission/controlled/*.npz sadproj",
        data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
