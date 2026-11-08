"""The scaling sign under the paper's folded statistic and under a signed band statistic.

Post-submission descriptive check (docs/post-submission-checks.md), not registered. The paper's
statistic folds each layer's AUROC (max(AUROC, 1 - AUROC)) and takes the maximum over all hidden
states, so a direction pointing the wrong way scores as well as one pointing the right way, and the
maximum can sit at the embedding output. The signed alternative is the mean over hidden states at
relative depth 0.25 to 0.75 (index / (number of hidden states - 1), hidden state 0 counted) of
(AUROC - 0.5), with AUROC oriented so that above 0.5 means the evaluation-minus-deployment
direction scores testing items higher.

For every ladder and rendering, every model and every one of the 36 full-task wrappers (scoring-set
projections stored in results/post_submission/controlled/<ladder>__<model>.npz, key sadproj): both
statistics; per wrapper, Pearson r with log10 parameters under both; sign counts, eval-fixed and
deploy-fixed spanning counts, and r of the 36-wrapper average under both.

    python scripts/signed_statistic.py
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
OUT = ROOT / "results" / "post_submission" / "signed_statistic.json"
BAND = (0.25, 0.75)


def per_model(path):
    with np.load(path, allow_pickle=False) as z:
        P, y = z["sadproj"].astype(np.float64), z["y"].astype(int)
        real, frs = z["real"], [str(f) for f in z["framings"]]
    n, L, J = P.shape
    a = auc_cols(y, P.reshape(n, L * J)).reshape(L, J)
    depth = np.arange(L) / (L - 1)
    band = (depth >= BAND[0]) & (depth <= BAND[1])
    fold = np.abs(np.maximum(a, 1.0 - a) - 0.5).max(0)
    signed = (a[band] - 0.5).mean(0)
    wrappers = [f"{frs[e]} x {frs[d]}" for e, d in real]
    return fold, signed, wrappers


def summary(lp, X, wrappers):
    """X [models, 36] -> per-wrapper r and the sign counts of pod_ladders."""
    r = np.array([np.corrcoef(lp, X[:, j])[0, 1] for j in range(X.shape[1])])
    ev = sorted({w.split(" x ")[0] for w in wrappers})
    dp = sorted({w.split(" x ")[1] for w in wrappers})
    by_e = [[j for j, w in enumerate(wrappers) if w.split(" x ")[0] == e] for e in ev]
    by_d = [[j for j, w in enumerate(wrappers) if w.split(" x ")[1] == d] for d in dp]
    return {"r": r.tolist(), "r_min": float(r.min()), "r_max": float(r.max()),
            "n_positive": int((r > 0).sum()), "n_negative": int((r < 0).sum()),
            "eval_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_e)),
            "deploy_fixed_spanning": int(sum(r[i].max() > 0 > r[i].min() for i in by_d)),
            "r_of_36_average": float(np.corrcoef(lp, X.mean(1))[0, 1])}


def main() -> None:
    out, pop = {}, []
    for ladder, (_, params) in LADDERS.items():
        models = list(params)
        lp = np.array([np.log10(params[m]) for m in models])
        F, S, W = [], [], None
        for m in models:
            f, s, w = per_model(CELLS / f"{ladder}__{m}.npz")
            assert W is None or w == W, "wrapper order differs between models"
            F.append(f), S.append(s)
            W = w
        F, S = np.array(F), np.array(S)
        sf, ss = summary(lp, F, W), summary(lp, S, W)
        rf, rs = np.array(sf["r"]), np.array(ss["r"])
        out[ladder] = {
            "models": models, "params_b": [params[m] for m in models], "wrappers": W,
            "fold": F.tolist(), "signed_band": S.tolist(),
            "share_wrapper_model_cells_signed_below_zero": float((S < 0).mean()),
            "scaling_fold": sf, "scaling_signed": ss,
            "r_fold_vs_r_signed_pearson": float(np.corrcoef(rf, rs)[0, 1]),
            "n_wrappers_same_sign_fold_and_signed": int((np.sign(rf) == np.sign(rs)).sum())}
        pop += models
        print(f"{ladder:16s} fold: r {sf['r_min']:+.2f}..{sf['r_max']:+.2f} +{sf['n_positive']}/-"
              f"{sf['n_negative']} span {sf['eval_fixed_spanning']}/{sf['deploy_fixed_spanning']} "
              f"r36 {sf['r_of_36_average']:+.2f} | signed: r {ss['r_min']:+.2f}..{ss['r_max']:+.2f} "
              f"+{ss['n_positive']}/-{ss['n_negative']} span {ss['eval_fixed_spanning']}/"
              f"{ss['deploy_fixed_spanning']} r36 {ss['r_of_36_average']:+.2f} | cells below 0 "
              f"{out[ladder]['share_wrapper_model_cells_signed_below_zero']:.2f} | same sign "
              f"{out[ladder]['n_wrappers_same_sign_fold_and_signed']}/36", flush=True)
    write_result(OUT, {"ladders": out, "band": list(BAND),
                       "check": "post-submission descriptive check, not registered "
                                "(docs/post-submission-checks.md)"},
                 kind="measured",
                 statistic="fold = the paper's max over all hidden states of |max(AUROC, 1-AUROC) - "
                           "0.5|; signed = mean over relative depth 0.25-0.75 of (AUROC - 0.5); "
                           "per-wrapper Pearson r with log10 parameters",
                 note="full-task directions of the paper; hidden state 0 counts in the depth",
                 pooling="validated (Qwen2.5: lf scoring, all-token directions, as pod_ladders)",
                 layers="ALL for fold; relative depth 0.25-0.75 for signed",
                 scoring_array="results/post_submission/controlled/*.npz sadproj",
                 data_root="pod + gstudy", model_population=sorted(set(pop)))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
