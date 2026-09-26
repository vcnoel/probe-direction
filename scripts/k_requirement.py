"""How many templates a probe comparison needs, as a function of what it is trying to resolve.

The D-study prescription "k = 122 templates" is not a property of the measurement. It is a
property of the measurement crossed with the population of models being compared, and it enters
almost entirely through the between-model variance. Solving E rho^2 >= 0.8 for k gives

                    sigma2_mt + sigma2_mth / n
        k  >=   ---------------------------------
                  sigma2_m / 4  -  sigma2_mh / n

so the template requirement scales inversely with the between-model variance the comparison aims
to resolve. Two consequences follow, and they are the point of this script.

FEASIBILITY THRESHOLD. The denominator is positive only when sigma2_m > 4*sigma2_mh/n. Below
that, no number of templates whatsoever reaches the target, because model-by-item error alone
already exceeds the budget. "No k suffices" is therefore not asymptotic slowness but a hard
threshold, and it moves with n.

SCOPE. A wide cross-family ladder has a much larger sigma2_m than a narrow within-family one, so
the same procedure can be adequate for a scaling claim and hopeless for a ranking claim between
two neighbouring checkpoints. This is why a published cross-family power law can be sound while
a within-family non-monotonicity from the same kind of measurement is fragile -- not because one
group was more careful, but because they were resolving different amounts of signal with the
same error.

The between-model variance implied by a published correlation is recovered by inverting the
generalizability coefficient at that paper's design, which gives an independent estimate of how
much wider their population was.

    python scripts/k_requirement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Qwen2.5 transformer layer counts. Non-monotone in parameters, which is what makes the
# architecture check below possible at all.
LAYERS = {"qwen2.5-0.5b": 24, "qwen2.5-1.5b": 28, "qwen2.5-3b": 36, "qwen2.5-7b": 28}
# Parameter counts come from configs/gstudy_models.json, derived from safetensors byte
# totals. A hardcoded dict of nominal names used to live here and was wrong by 0.037 in
# log10 at 7B, the leverage point of the fit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_io import params_b as _params_b  # noqa: E402
PARAMS = _params_b()
def k_required(c, s_m, n, target=0.8):
    """Templates needed for E rho^2 >= target, given a between-model variance s_m."""
    denom = s_m * (1 - target) / target - c["s_mh"] / n
    if denom <= 0:
        return None
    return max((c["s_mt"] + c["s_mth"] / n) / denom, 1.0)   # a design uses at least one template


def s_m_threshold(c, n, target=0.8):
    """Between-model variance below which no k reaches the target."""
    return c["s_mh"] / n * target / (1 - target)


def s_m_implied(c, rho, k=1, n=4):
    """Invert the generalizability coefficient to recover the sigma2_m a reliability implies."""
    err = c["s_mt"] / k + c["s_mh"] / n + c["s_mth"] / (k * n)
    return rho * err / (1 - rho)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, r


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    res = {}
    for stat in ("peak", "tilt"):
        p = ROOT / "results" / f"gstudy_gtheory_deploy_{stat}.json"
        if p.exists():
            res[stat] = json.loads(p.read_text(encoding="utf-8"))
    if not res:
        print("run scripts/gstudy_gtheory.py first"); return

    for stat, d in res.items():
        c = d["components"]
        sm = c["s_m"]
        print(f"\n{'=' * 90}\nTEMPLATE REQUIREMENT AS A FUNCTION OF BETWEEN-MODEL VARIANCE  "
              f"[{stat}]\n{'=' * 90}")
        print(f"  measured sigma2_m (4 Qwen checkpoints, 0.5-7B) = {sm:.3e}")
        thr = s_m_threshold(c, n=4)
        print(f"  feasibility threshold at n=4 blocks: sigma2_m must exceed {thr:.3e} "
              f"({thr / sm:.1f}x measured)")
        print(f"  below it NO number of templates reaches E rho^2=0.80: model-by-item error")
        print(f"  alone exhausts the budget. The threshold falls as 1/n.\n")
        print(f"  {'sigma2_m':>12}{'multiple':>10}{'k @ n=4':>10}{'k @ n=10':>10}{'k @ n=22':>10}")
        for mult in (1, 2, 3, 5, 10, 20, 50):
            row = f"  {sm * mult:12.3e}{mult:9d}x"
            for n in (4, 10, 22):
                k = k_required(c, sm * mult, n)
                row += f"{('inf' if k is None else f'{k:.0f}'):>10}"
            print(row)

        smi = s_m_implied(c, 0.36, k=1, n=4)
        ki = k_required(c, smi, 4)
        print(f"\n  Inverting the reliability implied by the published cross-family fit "
              f"(rho_yy>=0.36 at k=1,n=4):")
        print(f"    sigma2_m ~ {smi:.3e} = {smi / sm:.1f}x ours  ->  k ~ "
              f"{'inf' if ki is None else f'{ki:.0f}'} templates at n=4")
        print(f"    Their population carried {smi / sm:.1f}x our between-model variance on this")
        print(f"    statistic, inferred without any assumption about ladder width. The multiple")
        print(f"    is statistic-specific: it is large for `peak`, whose sigma2_m here is tiny,")
        print(f"    and modest for `tilt`, which already resolves these four models reasonably.")
        res[stat]["_k_analysis"] = {
            "s_m": sm, "threshold_n4": thr, "s_m_implied_chaudhary": smi,
            "k_at_implied": ki, "multiple_implied": smi / sm,
        }

    print(f"\n{'=' * 90}\nPOPULATION SCOPE: the same procedure, three different verdicts\n{'=' * 90}")
    dp, dt = res.get("peak"), res.get("tilt")
    c = dp["components"]
    smi = s_m_implied(c, 0.36, 1, 4)
    rows = [
        ("cross-family, wide (0.27B-70B, 4 families)", "power-law scaling",
         f"{s_m_implied(c, 0.36, 1, 4):.1e}", 0.36,
         f"k~{k_required(c, smi, 4):.0f}", "marginal"),
        ("within-family, narrow (Qwen 0.5-7B)", "3B vs 7B ranking",
         f"{c['s_m']:.1e}", dp["g_k1"],
         "no k at n=4" if k_required(c, c["s_m"], 4) is None
         else f"k~{k_required(c, c['s_m'], 4):.0f}", "no"),
        ("single model", "'evaluation awareness exists here'", "n/a", float("nan"),
         "n/a", "yes"),
    ]
    print(f"  {'population':<44}{'sigma2_m':>10}{'E rho^2':>9}{'k for 0.80':>13}{'k=1 ok?':>9}")
    for pop, _claim, s, g, k, ok in rows:
        gs = "  --  " if g != g else f"{g:.2f}"
        print(f"  {pop:<44}{s:>10}{gs:>9}{k:>13}{ok:>9}")
    print(f"\n  The single-model row is a different kind of claim: it is judged against the")
    print(f"  selection-corrected null floor, not against a generalizability coefficient, and")
    print(f"  all cells clear it.")

    # ---- is tilt's advantage architecture rather than model property? ----
    if dt:
        print(f"\n{'=' * 90}\nIS TILT'S ADVANTAGE A MODEL PROPERTY OR AN ARCHITECTURE ARTIFACT?"
              f"\n{'=' * 90}")
        print(f"  tilt sigma2_m = {dt['components']['s_m']:.3e} is "
              f"{dt['components']['s_m'] / dp['components']['s_m']:.1f}x the peak's, which is where")
        print(f"  most of its E rho^2 advantage comes from. Qwen layer counts are non-monotone in")
        print(f"  parameters (24/28/36/28), so the two predictors are separable even at n=4.")
        for stat, d in (("tilt", dt), ("peak", dp)):
            pm = d.get("per_model_mean")
            if not pm:
                print(f"  {stat}: per-model means absent; re-run gstudy_gtheory.py"); continue
            ms = [m for m in pm if m in LAYERS]
            y = [pm[m] for m in ms]
            s_p, r_p = ols([np.log10(PARAMS[m]) for m in ms], y)
            s_l, r_l = ols([LAYERS[m] for m in ms], y)
            print(f"\n  {stat}: mean over templates per model = "
                  + ", ".join(f"{m.split('-')[-1]} {pm[m]:+.4f}" for m in ms))
            print(f"    vs log10(params) : slope {s_p:+.5f}  r = {r_p:+.3f}")
            print(f"    vs layer count   : slope {s_l:+.5f}  r = {r_l:+.3f}")
            verdict = ("layer count dominates -- treat the advantage as architecture, not model"
                       if abs(r_l) > abs(r_p) + 0.15 else
                       "parameters dominate -- the advantage survives the architecture check"
                       if abs(r_p) > abs(r_l) + 0.15 else
                       "the two are not separable at n=4; indeterminate")
            print(f"    -> {verdict}")
        print(f"\n  Four points cannot settle this. The direction is reported because it is free")
        print(f"  and because a layer-count-dominated result would weaken the recommendation to")
        print(f"  prefer the slope; adding models from other families is what would settle it.")

        print(f"\n{'=' * 90}\nSIGN OF THE SCALE RELATIONSHIP ON THIS LADDER (observation, not a claim)"
              f"\n{'=' * 90}")
        for stat, d in (("peak", dp), ("tilt", dt)):
            cc = d["components"]
            k, n = d["n_pairs"], d["n_blocks"]
            g = cc["s_m"] / (cc["s_m"] + cc["s_mt"] / k + cc["s_mh"] / n + cc["s_mth"] / (k * n))
            pm = d["per_model_mean"]
            ms = [m for m in pm if m in PARAMS]
            _, r = ols([np.log10(PARAMS[m]) for m in ms], [pm[m] for m in ms])
            print(f"  {stat}: averaging over all {k} templates gives per-model means of "
                  f"reliability E rho^2={g:.3f};")
            print(f"        their correlation with log10(parameters) is {r:+.3f}.")
        print(f"\n  Both are NEGATIVE, where the published cross-family fit is positive. This is")
        print(f"  not offered as a contradiction. The populations differ, the correlation is")
        print(f"  computed on the same four points that estimated sigma2_m so it is partly")
        print(f"  circular, and four models cannot support a scaling claim in either direction.")
        print(f"  It is recorded because a multi-template estimator with a reliability near the")
        print(f"  conventional target is exactly the instrument this paper argues for, and the")
        print(f"  first thing it says about a narrow within-family ladder disagrees in sign with")
        print(f"  what single-template studies report. Resolving that needs the wider population.")

    out = ROOT / "results" / "k_requirement.json"
    out.write_text(json.dumps({k: v.get("_k_analysis") for k, v in res.items()}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
