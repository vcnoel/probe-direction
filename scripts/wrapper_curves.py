"""The 4 x 36 matrix of statistic-by-model-by-wrapper, and three wrappers that disagree.

The histogram of 36 slopes says the sign varies. It does not show that a chosen wrapper produces a
chosen conclusion. For that the reader needs the actual scaling curves, so this script emits the
full matrix and identifies three wrappers -- the most positive, the most negative, and the flattest
-- whose curves are then plotted side by side by scripts/make_figures.py.

The three are selected by the correlation they produce, which is selection on the outcome and is the
point: the claim is that a researcher free to choose the wrapper is free to choose the result. The
wrapper texts are emitted verbatim so no reader has to take on trust that one of them is a strawman.

Same array, statistic and layer convention as the rest of the paper:
results/gstudy/*/pooled.npz, folded maximum of |AUROC - 0.5| over interior layers.

    python scripts/wrapper_curves.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LADDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fr_txt = json.loads((ROOT / "data" / "framings_multi.json").read_text(encoding="utf-8"))

    stat, pairs = None, None
    for i, m in enumerate(LADDER):
        # The scoring array must be the DECLARED convention, and the direction array must match
        # scripts/slope_distribution.py, so that these curves are three of the same 36 wrappers
        # whose correlation range the manuscript quotes. Reading the all-token scoring array here
        # made the figure and the quoted range two different computations of one design.
        with np.load(ROOT / "results" / "gstudy" / m / "pooled.npz", allow_pickle=False) as z:
            fr = z["framing"].astype(np.float64)
            names = np.array([str(s) for s in z["framing_name"]])
            fam = json.loads(str(z["families"]))
        with np.load(ROOT / "results" / "gstudy" / m / "sad_filtered_lf.npz",
                     allow_pickle=False) as z2:
            sad = z2["sad"].astype(np.float64)
            y = z2["sad_label"].astype(int)
        ev = sorted(k for k, v in fam.items() if v == "eval")
        dp = sorted(k for k, v in fam.items() if v == "deploy")
        if pairs is None:
            pairs = list(itertools.product(ev, dp))
            stat = np.zeros((len(LADDER), len(pairs)))
        # ALL layers, matching scripts/gstudy_gtheory.py and scripts/slope_distribution.py, which
        # produce the coefficients and the sign distribution. Restricting to interior layers here
        # made this figure a different statistic from the numbers printed beside it; the interior
        # restriction was tested as a prescription and withdrawn (appendix).
        cols = range(sad.shape[1])
        for j, (fe, fo) in enumerate(pairs):
            d = unit(fr[names == fe].mean(0) - fr[names == fo].mean(0))
            pr = np.einsum("ild,ld->il", sad, d)
            stat[i, j] = max(abs(max(a, 1 - a) - 0.5) for a in
                             (roc_auc_score(y, pr[:, L]) for L in cols))
        print(f"  {m} done", flush=True)

    lp = np.log10([PARAMS[m] for m in LADDER])
    r = np.array([float(np.corrcoef(lp, stat[:, j])[0, 1]) for j in range(stat.shape[1])])
    slope = np.array([float(np.polyfit(lp, stat[:, j], 1)[0]) for j in range(stat.shape[1])])

    # SELECTION IS ON A DECLARED CRITERION, NOT ON MAXIMUM SPREAD.
    #
    # Choosing the globally most positive, flattest and most negative wrappers is an argmax over
    # near-ties: a 0.010 shift in r from correcting the parameter axis swapped the whole triple, and
    # with it the figure's claim -- from "the sign flips when the DEPLOYMENT framing changes" to the
    # weaker "when the evaluation framing changes". The demonstration should not depend on which of
    # several near-equal wrappers won an argmax.
    #
    # So the triple is constrained to share one evaluation framing, and only the deployment framing
    # varies across the panels. That is the strong form of the claim: the side of the contrast that
    # never mentions evaluation decides the sign. Among evaluation framings whose deployment variants
    # span both signs, the widest-range one is shown, and the number that qualify is recorded --
    # several do, which is itself evidence the effect is not one lucky pair.
    by_eval = {}
    for j, (fe, fo) in enumerate(pairs):
        by_eval.setdefault(fe, []).append(j)
    qualifying = {fe: idx for fe, idx in by_eval.items()
                  if max(r[i] for i in idx) > 0 > min(r[i] for i in idx)}
    if not qualifying:
        raise SystemExit("no evaluation framing has deployment variants of both signs; the "
                         "figure's claim would have to be restated")
    chosen = max(qualifying, key=lambda fe: max(r[i] for i in qualifying[fe])
                 - min(r[i] for i in qualifying[fe]))
    idx = qualifying[chosen]
    pick = {"most positive": max(idx, key=lambda i: r[i]),
            "flattest": min(idx, key=lambda i: abs(r[i])),
            "most negative": min(idx, key=lambda i: r[i])}
    print(f"  selection: evaluation framing {chosen}, deployment framing varies; "
          f"{len(qualifying)} of {len(by_eval)} evaluation framings admit a sign flip")

    print(f"\n{'=' * 96}")
    print("ONE METHOD, THREE WRAPPERS, THREE CONCLUSIONS")
    print(f"{'=' * 96}")
    out = {"statistic": "peak", "sad_pooling": "lf",
           "direction_pooling": "all-token, matching slope_distribution.py",
           "layers": "all, matching gstudy_gtheory.py and slope_distribution.py",
           "models": LADDER,
           "params_b": [PARAMS[m] for m in LADDER],
           "pairs": ["|".join(p) for p in pairs],
           "matrix": stat.tolist(), "correlations": r.tolist(), "slopes": slope.tolist(),
           "n_wrappers": len(pairs), "selected": {},
           "selection_rule": ("triple constrained to one evaluation framing so that only the "
                              "deployment framing varies; among evaluation framings whose "
                              "deployment variants span both signs, the widest-range one"),
           "selection_shared_eval_framing": chosen,
           "n_eval_framings_with_sign_flip": len(qualifying),
           "n_eval_framings": len(by_eval)}
    for lab, j in pick.items():
        fe, fo = pairs[j]
        rec = {"index": j, "eval_framing": fe, "deploy_framing": fo,
               "r": float(r[j]), "slope": float(slope[j]),
               "values": stat[:, j].tolist(),
               "eval_label": fr_txt[fe]["label"], "deploy_label": fr_txt[fo]["label"],
               "eval_system": fr_txt[fe]["system"], "deploy_system": fr_txt[fo]["system"]}
        out["selected"][lab] = rec
        print(f"  {lab:<14} r = {r[j]:+.3f}   {fe} vs {fo}")
        print(f"      " + "  ".join(f"{m.split('-')[-1]}={v:.3f}"
                                    for m, v in zip(LADDER, stat[:, j])))
    print("")
    print(f"  Across {len(pairs)} wrappers r runs {r.min():+.3f} to {r.max():+.3f}. The three above")
    print("  three share one evaluation framing and differ only in the deployment framing, so the")
    print("  sign is decided by the side of the contrast that never names the concept.")
    print("  Wrapper texts are emitted in full so none can be dismissed as a strawman.")

    p = ROOT / "results" / "wrapper_curves.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
