"""Where the paper's statistic is read: the hidden state at which the folded maximum sits.

Post-submission descriptive check (docs/post-submission-checks.md), not registered. The statistic
is the maximum over ALL hidden states of |AUROC - 0.5|; hidden state 0 is the embedding output, so
a maximum there is a lexical read-out, not a representation built by the network. For every model
of every ladder and rendering, from the scoring-set projections of the 36 full-task directions
stored by scripts/controlled_transfer.py (results/post_submission/controlled/<ladder>__<model>.npz,
key sadproj), this reports the share of wrappers whose maximum sits at hidden state 0 exactly, in
the first 10 percent of depth (index / (number of hidden states - 1) <= 0.10), in the last 10
percent (>= 0.90), and the median relative depth of the maximum.

    python scripts/read_layer.py
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
OUT = ROOT / "results" / "post_submission" / "read_layer.json"


def curves(path):
    """-> folded |AUROC - 0.5| per hidden state and wrapper [L, 36], the stored full-task peak."""
    with np.load(path, allow_pickle=False) as z:
        P, y, peak = z["sadproj"].astype(np.float64), z["y"].astype(int), z["full_peak"]
    n, L, J = P.shape
    a = auc_cols(y, P.reshape(n, L * J)).reshape(L, J)
    return np.abs(np.maximum(a, 1.0 - a) - 0.5), peak


def main() -> None:
    out, pop = {}, []
    for ladder, (_, params) in LADDERS.items():
        rows = []
        for model in params:
            c, stored = curves(CELLS / f"{ladder}__{model}.npz")
            # sadproj is stored in float32; the stored peak came from float64 projections
            assert np.abs(c.max(0) - stored).max() < 2e-3, f"{ladder}/{model}: peak mismatch"
            L = c.shape[0]
            at = c.argmax(0)
            depth = at / (L - 1)
            rows.append({"model": model, "n_hidden_states": L,
                         "share_at_state_0": float((at == 0).mean()),
                         "share_first_10pct": float((depth <= 0.10).mean()),
                         "share_last_10pct": float((depth >= 0.90).mean()),
                         "median_relative_depth": float(np.median(depth)),
                         "argmax_state": at.tolist()})
            print(f"{ladder:16s} {model:26s} L={L:3d}  state 0 {rows[-1]['share_at_state_0']:.2f}  "
                  f"first 10% {rows[-1]['share_first_10pct']:.2f}  last 10% "
                  f"{rows[-1]['share_last_10pct']:.2f}  median depth "
                  f"{rows[-1]['median_relative_depth']:.2f}", flush=True)
        out[ladder] = rows
        pop += list(params)
    write_result(OUT, {"ladders": out, "n_wrappers": 36,
                       "check": "post-submission descriptive check, not registered "
                                "(docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="argmax over hidden states of the folded |AUROC - 0.5| of each full-task "
                           "direction on the SAD scoring set (the paper's statistic is its maximum)",
                 note="hidden state 0 is the embedding output; relative depth = index / (number of "
                      "hidden states - 1)",
                 pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="ALL", scoring_array="results/post_submission/controlled/*.npz sadproj",
                 data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
