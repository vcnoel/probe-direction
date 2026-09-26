"""Do wrappers DETERMINE the sign, or is the sign merely unestimable at four models?

Both hypotheses predict the observation the paper leads with. For every evaluation framing, some
deployment framing reverses the sign of the scaling relationship -- 6 of 6 -- but under a null where
each wrapper's r is noise symmetric about zero, one framing's six deployment variants span both signs
with probability 1 - 2(1/2)^6 = 0.969, and all six framings do so with probability 0.969^6 = 0.83.
So 6 of 6 is nearly guaranteed by noise. It refutes "the sign is a stable property of the model",
which the paper needs, and it does not establish "the wrapper determines the sign", which the paper
currently asserts in Section 4's opening sentence and in Figure 1's caption.

The discriminating measurement is the reliability of the per-wrapper slope. Split the scoring items
into disjoint halves, recompute each wrapper's slope on each half independently, and correlate the two
sets of 36 slopes. A wrapper whose slope is a real property of that wrapper gives the same slope on
either half; a slope that is sampling noise does not.

  reliable   -> the sign is a property of the wrapper, and "determines" is the right word
  unreliable -> the sign is unestimable at this design, and the honest claim is that it is not
                identified, which still refutes any per-model reading

The verdict depends on Spearman-Brown and nothing else. That matters, because the first version of
this script ALSO gated on pooled sign agreement at 0.75, reported "partially reliable" when that
came back 0.700, and was then changed to gate on top-quarter agreement at 0.90 -- a threshold whose
quantile and cutoff were both chosen with the answer already on screen. The slope numbers never
moved; only the gate did. That is the paper's own thesis applied to the paper, and the gate is gone.

The magnitude-conditional agreements are still computed and printed, because a reader should see
that near-zero slopes disagree in sign more often than large ones. They are descriptive. They cannot
license anything, and nothing in the manuscript may rest on them: sign is a deterministic function
of slope, so sign instability at small magnitude is arithmetic, not a second finding. The finding is
that the slope reproduces on held-out items.

Same array, statistic and layer convention as the rest of the paper.

    python scripts/slope_reliability.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import params_b  # noqa: E402

LADDER = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
N_SPLITS = 20


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def peak(y, pr, cols):
    return max(abs(max(a, 1 - a) - 0.5) for a in
               (roc_auc_score(y, pr[:, L]) for L in cols))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    PARAMS = params_b()
    proj, y_ref, pairs, cols = {}, None, None, None
    for m in LADDER:
        with np.load(ROOT / "results" / "gstudy" / m / "pooled.npz", allow_pickle=False) as z:
            fr = z["framing"].astype(np.float64)
            names = np.array([str(s) for s in z["framing_name"]])
            fam = json.loads(str(z["families"]))
        with np.load(ROOT / "results" / "gstudy" / m / "sad_filtered_lf.npz",
                     allow_pickle=False) as z2:
            sad = z2["sad"].astype(np.float64)
            y = z2["sad_label"].astype(int)
        if pairs is None:
            ev = sorted(k for k, v in fam.items() if v == "eval")
            dp = sorted(k for k, v in fam.items() if v == "deploy")
            pairs = list(itertools.product(ev, dp))
            y_ref, cols = y, range(sad.shape[1])
        for j, (fe, fo) in enumerate(pairs):
            d = unit(fr[names == fe].mean(0) - fr[names == fo].mean(0))
            proj[(m, j)] = np.einsum("ild,ld->il", sad, d)
        print(f"  {m} projected", flush=True)

    lp = np.array([np.log10(PARAMS[m]) for m in LADDER])
    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y_ref == 1), np.flatnonzero(y_ref == 0)
    r_corrs, s_corrs, sign_agree = [], [], []
    agree_all, agree_top50, agree_top25 = [], [], []

    for s in range(N_SPLITS):
        pa = rng.permutation(pos); na = rng.permutation(neg)
        h1 = np.concatenate([pa[:len(pa) // 2], na[:len(na) // 2]])
        h2 = np.concatenate([pa[len(pa) // 2:], na[len(na) // 2:]])
        slopes, rs = {}, {}
        for half, idx in (("a", h1), ("b", h2)):
            st = np.array([[peak(y_ref[idx], proj[(m, j)][idx], cols) for j in range(len(pairs))]
                           for m in LADDER])
            slopes[half] = np.array([np.polyfit(lp, st[:, j], 1)[0] for j in range(len(pairs))])
            rs[half] = np.array([np.corrcoef(lp, st[:, j])[0, 1] for j in range(len(pairs))])
        # Sign agreement conditional on magnitude. A wrapper whose slope is near zero has no sign
        # to reproduce, so pooling it with the extremes understates how stable the strong ones are --
        # and the extremes are what the figure shows.
        mag = np.abs((slopes["a"] + slopes["b"]) / 2)
        for q, store in ((0.0, agree_all), (0.5, agree_top50), (0.75, agree_top25)):
            keep = mag >= np.quantile(mag, q)
            store.append(float(np.mean(np.sign(slopes["a"][keep]) == np.sign(slopes["b"][keep]))))
        s_corrs.append(float(np.corrcoef(slopes["a"], slopes["b"])[0, 1]))
        r_corrs.append(float(np.corrcoef(rs["a"], rs["b"])[0, 1]))
        sign_agree.append(float(np.mean(np.sign(slopes["a"]) == np.sign(slopes["b"]))))
        print(f"    split {s + 1}/{N_SPLITS}  slope r={s_corrs[-1]:+.3f}  "
              f"sign agreement {sign_agree[-1]:.2f}", flush=True)

    sc, rc, sa = np.array(s_corrs), np.array(r_corrs), np.array(sign_agree)
    # Spearman-Brown: reliability of the FULL-length measurement from a half-length correlation.
    sb = 2 * sc.mean() / (1 + sc.mean())
    a50, a25 = np.array(agree_top50), np.array(agree_top25)

    print(f"\n{'=' * 96}")
    print("IS THE PER-WRAPPER SLOPE A PROPERTY OF THE WRAPPER, OR SAMPLING NOISE?")
    print(f"{'=' * 96}")
    print(f"  {N_SPLITS} disjoint item splits, {len(pairs)} wrappers, {len(LADDER)} models.")
    print(f"  slope split-half correlation   {sc.mean():+.3f}  (sd {sc.std(ddof=1):.3f}, "
          f"range {sc.min():+.3f} to {sc.max():+.3f})")
    print(f"  Spearman-Brown full-length     {sb:+.3f}")
    print(f"  r split-half correlation       {rc.mean():+.3f}")
    print(f"  sign agreement, all wrappers   {sa.mean():.3f}  (chance 0.5)")
    print(f"  sign agreement, top half by |slope|    {a50.mean():.3f}   descriptive only")
    print(f"  sign agreement, top quarter by |slope| {a25.mean():.3f}   descriptive only")
    print("")
    # The verdict turns on Spearman-Brown and nothing else, at the 0.7 this script was written with.
    # The magnitude-conditional agreements below it are reported because they are informative, and
    # they license nothing: see the note in the docstring.
    if sb >= 0.7:
        verdict = ("reliable: the per-wrapper slope reproduces on independent items, so the slope "
                   "is a property of the wrapper and 'determines' is the right word")
    elif sb >= 0.4:
        verdict = ("partially reliable: the slope carries real wrapper-specific signal but not "
                   "enough to call it a property of the wrapper; 'is not identified' is the "
                   "defensible claim and any stronger wording needs this number quoted beside it")
    else:
        verdict = ("unreliable: the per-wrapper slope does not reproduce on independent items, so "
                   "the sign is unestimable at this design rather than wrapper-determined; every "
                   "'determines' and 'chooses the conclusion' must be softened")
    print(f"  VERDICT  {verdict}")
    print("")
    print("  Either verdict refutes a per-MODEL reading of the sign, which is what the reanalysis")
    print("  needs; what is at stake here is only whether the wrapper is the positive cause.")

    out = {"n_splits": N_SPLITS, "n_wrappers": len(pairs), "models": LADDER,
           "slope_split_half_mean": float(sc.mean()), "slope_split_half_sd": float(sc.std(ddof=1)),
           "spearman_brown": float(sb), "r_split_half_mean": float(rc.mean()),
           "sign_agreement_mean": float(sa.mean()),
           "sign_agreement_top50": float(a50.mean()),
           "sign_agreement_top25": float(a25.mean()),
           "null_prob_6of6": 0.969 ** 6, "verdict": verdict}
    p = ROOT / "results" / "slope_reliability.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
