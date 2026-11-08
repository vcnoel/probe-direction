"""The paper's statistic unfolded at its own read-out state, with the layer set held fixed.

Post-submission descriptive check (docs/post-submission-checks.md), not registered; first computed
by the adversarial audit. signed_statistic.py changed two things at once: it unfolds the AUROC and
moves from the maximum over all hidden states to a mean over the interior band. This isolates the
fold. For every model and wrapper, the state at which the paper's folded statistic is maximal is
kept, and the AUROC there is read with its sign (AUROC - 0.5; below zero means the
evaluation-minus-deployment direction scores deployment items higher). Reported per ladder and
rendering: the share of model-by-wrapper cells below zero, and the 36-wrapper-average scaling r
and per-wrapper sign counts, folded against unfolded. Done for the submitted set (all hidden
states) and for the released set (states 1..n-2).

    python scripts/unfold_isolate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pod_ladders import LADDERS  # noqa: E402
from released_layers import auroc_curves, layer_sets, scaling  # noqa: E402
from result_io import write_result  # noqa: E402

OUT = ROOT / "results" / "post_submission" / "unfold_isolate.json"
SETS = ("all", "released")


def main() -> None:
    out, pop = {}, []
    for ladder, (_, params) in LADDERS.items():
        models = list(params)
        lp = np.array([np.log10(params[m]) for m in models])
        fold = {s: [] for s in SETS}
        unfold = {s: [] for s in SETS}
        real = None
        for m in models:
            a, real = auroc_curves(ladder, m)
            J = a.shape[1]
            for s in SETS:
                lay = layer_sets(a.shape[0])[s]
                f = np.abs(a[lay] - 0.5)
                at = f.argmax(0)                                  # the paper's read-out state
                fold[s].append(f[at, np.arange(J)])
                unfold[s].append(a[lay][at, np.arange(J)] - 0.5)
        rec = {}
        for s in SETS:
            F, U = np.array(fold[s]), np.array(unfold[s])
            assert np.allclose(np.abs(U), F)
            rec[s] = {"share_cells_below_zero": float((U < 0).mean()),
                      "share_cells_below_zero_per_model": (U < 0).mean(1).tolist(),
                      "folded": scaling(lp, F, real), "unfolded": scaling(lp, U, real)}
        out[ladder] = {"models": models, "by_layer_set": rec}
        pop += models
        print(f"{ladder:16s} " + " | ".join(
            f"{s}: below 0 {rec[s]['share_cells_below_zero']:.2f}  r36 folded "
            f"{rec[s]['folded']['r_of_36_average']:+.2f} -> unfolded "
            f"{rec[s]['unfolded']['r_of_36_average']:+.2f}  unfolded +{rec[s]['unfolded']['n_positive']}"
            f"/-{rec[s]['unfolded']['n_negative']}" for s in SETS), flush=True)
    write_result(OUT, {"ladders": out,
                       "check": "post-submission descriptive check, not registered; first computed "
                                "by the adversarial audit (docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="AUROC - 0.5 at the hidden state where the paper's folded |AUROC - 0.5| "
                           "is maximal, against the folded value; 36-average and per-wrapper "
                           "Pearson r with log10 parameters",
                 note="all = every hidden state (submitted); released = states 1..n-2",
                 pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="all hidden states; states 1..n-2",
                 scoring_array="results/post_submission/controlled/*.npz sadproj",
                 data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
