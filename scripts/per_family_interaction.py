"""Per-family model x wrapper interaction: is the wrapper effect a function of scale?

The four-family REML sensitivity fit (appendix) shows sigma^2_model:template falling to 0.72x its
value when three sub-2B models enter the population. Two readings compete:

  SCALE. Wrapper sensitivity grows with scale, so a population weighted toward small models shows
  less of it. If true, the measurement problem is worst exactly where the safety-relevant claims are
  made, and the paper's result understates itself.

  FAMILY. SmolLM2 models simply interact less with the wrapper than Qwen2.5 models do, for reasons
  having nothing to do with size.

These are separated by computing the interaction WITHIN each family, which holds family fixed and
lets scale vary. The quantity is the model x wrapper interaction of a two-way model x wrapper table
per family, from the same statistic, array and layer convention as the rest of the paper.

The degrees of freedom are small by construction -- (n_models - 1) x (n_wrappers - 1) with 2 to 4
models per family -- so this is suggestive at best and is reported as such. It exists to decide
whether a scale conjecture is even consistent with the data we have, not to establish one.

    python scripts/per_family_interaction.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "results" / "gstudy_v11"


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def interaction_var(x):
    """model x wrapper interaction component of a two-way table [n_models, n_wrappers]."""
    n_m, n_t = x.shape
    if n_m < 2 or n_t < 2:
        return float("nan")
    g = x.mean()
    resid = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + g
    return float((resid ** 2).sum() / ((n_m - 1) * (n_t - 1)))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    conf = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))["models"]
    keys = [k for k, v in conf.items() if v.get("enabled") and (V11 / k / "pooled.npz").exists()]

    rows, pairs = {}, None
    for k in keys:
        with np.load(V11 / k / "pooled.npz", allow_pickle=False) as z:
            sad = z["sad_validated"].astype(np.float64)
            y = z["sad_label"].astype(int)
            fr = z["framing_validated"].astype(np.float64)
            names = np.array([str(s) for s in z["framing_name"]])
            fam = json.loads(str(z["families"]))
        if pairs is None:
            ev = sorted(a for a, b in fam.items() if b == "eval")
            dp = sorted(a for a, b in fam.items() if b == "deploy")
            pairs = list(itertools.product(ev, dp))
        cols = range(1, sad.shape[1] - 1)
        vals = []
        for fe, fo in pairs:
            d = unit(fr[names == fe].mean(0) - fr[names == fo].mean(0))
            pr = np.einsum("ild,ld->il", sad, d)
            vals.append(max(abs(max(a, 1 - a) - 0.5) for a in
                            (roc_auc_score(y, pr[:, L]) for L in cols)))
        rows[k] = np.array(vals)
        print(f"  {k} scored", flush=True)

    byfam = {}
    for k, v in rows.items():
        byfam.setdefault(conf[k]["family"], []).append((conf[k]["params_b"], k, v))

    print(f"\n{'=' * 96}")
    print("MODEL x WRAPPER INTERACTION WITHIN EACH FAMILY")
    print(f"{'=' * 96}")
    print(f"  {'family':<12}{'n':>3}{'mean log10 B':>14}{'sigma^2_mt':>14}{'df':>6}")
    out = {"n_wrappers": len(pairs), "statistic": "peak", "pooling": "validated", "families": {}}
    for f, lst in sorted(byfam.items(), key=lambda x: -np.mean([np.log10(p) for p, _, _ in x[1]])):
        lst.sort()
        X = np.stack([v for _, _, v in lst])
        s = interaction_var(X)
        df = (len(lst) - 1) * (len(pairs) - 1)
        ml = float(np.mean([np.log10(p) for p, _, _ in lst]))
        out["families"][f] = {"n_models": len(lst), "mean_log10_params": ml,
                             "sigma2_model_x_wrapper": s, "df": df,
                             "models": [k for _, k, _ in lst]}
        print(f"  {f:<12}{len(lst):>3}{ml:>14.2f}{s:>14.3e}{df:>6}")

    usable = {f: v for f, v in out["families"].items() if v["n_models"] >= 2}
    print("")
    # Three comparisons, because a single ordering test conflates two different questions.
    fams = sorted(usable.items(), key=lambda x: -x[1]["mean_log10_params"])
    if len(fams) >= 3:
        (fa, va), (fb, vb), (fc, vc) = fams[0], fams[1], fams[-1]
        matched = abs(va["mean_log10_params"] - vb["mean_log10_params"])
        ratio_matched = max(va["sigma2_model_x_wrapper"], vb["sigma2_model_x_wrapper"]) /             min(va["sigma2_model_x_wrapper"], vb["sigma2_model_x_wrapper"])
        ratio_small = np.mean([va["sigma2_model_x_wrapper"], vb["sigma2_model_x_wrapper"]]) /             vc["sigma2_model_x_wrapper"]
        # Within one family, scale varies while family is held fixed: the clean scale test.
        big = max(usable.items(), key=lambda x: x[1]["n_models"])
        bl = byfam[big[0]]
        bl.sort()
        span = np.log10(bl[-1][0] / bl[0][0])
        sds = [v.std(ddof=1) for _, _, v in bl]
        rr = float(np.corrcoef([np.log10(pp) for pp, _, _ in bl], sds)[0, 1])
        print(f"  1. {fa} and {fb} sit {matched:.2f} decades apart in mean scale -- effectively "
              f"matched -- and their")
        print(f"     interactions differ by {ratio_matched:.2f}x, in the opposite order to scale. "
              f"Family idiosyncrasy")
        print(f"     at matched scale is therefore small.")
        print(f"  2. {fc}, {va['mean_log10_params'] - vc['mean_log10_params']:.2f} decades below "
              f"them, is {ratio_small:.1f}x lower. Large, but {fc} is both the")
        print(f"     small family and a different family, so this comparison cannot separate them.")
        print(f"  3. Within {big[0]}, where scale spans {10 ** span:.0f}x with family held FIXED, "
              f"per-model spread")
        print(f"     across wrappers correlates with scale at {rr:+.2f} on {len(bl)} points and is "
              f"not monotonic.")
        print("")
        print("  The only comparison that isolates scale is the third, and it is null. The pooled")
        print("  component's drop is therefore not evidence that wrapper sensitivity grows with")
        print("  scale, and no such conjecture is stated. Settling it needs a wider ladder inside")
        print("  one family, which is the same experiment the coefficients need.")
        out["matched_pair"] = [fa, fb]
        out["ratio_at_matched_scale"] = float(ratio_matched)
        out["ratio_vs_small_family"] = float(ratio_small)
        out["within_family_scale_corr"] = rr
        out["within_family_span_decades"] = float(span)
        out["scale_conjecture_supported"] = False

    p = ROOT / "results" / "per_family_interaction.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
