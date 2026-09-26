"""Is the observed template concordance consistent with a stable model ranking?

Testing Kendall's W against the null of independent orderings asks the wrong question. That
null is a straw man: nobody claims templates rank models at random, and failing to reject it
says only that the design is small. The field's actual assumption is the opposite one -- that
the model ranking is a stable property and a template is an interchangeable measuring device --
and that assumption implies a PREDICTION about how much templates should agree.

So the reference distribution is built from that assumption instead. Fix one template and
resample items; each resample is a pseudo-judge that differs from the others by item noise
alone, which is exactly the world in which the ranking is a model property. The concordance
among those pseudo-judges is the W the field's assumption predicts. If the W observed across
real templates sits far below it, the conclusion is positive and directional -- observed
concordance is inconsistent with a stable model ranking -- and it does not depend on rejecting
independence.

Two further checks share the machinery.

Per-model template-averaged means. The published single-template values across these four
checkpoints span about 0.102. If averaging over templates compresses that span, most of the
apparent between-model variation in published work is model-by-template rather than model.

Between-model variance per implementation. sigma^2_m came out 2.0e-5 in the three-way design
at one implementation and 1.4e-4 in the four-way averaged over nine, a factor of seven. Either
the apparent between-model variance itself depends on the implementation, which is a sharp
claim, or these are unstable near-boundary estimates on three degrees of freedom, which is a
caveat. Computing it separately at each implementation distinguishes them.

    python scripts/concordance_equivalence.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_analyze import load, unit  # noqa: E402
from gstudy_gtheory import SADFILE, auc_cols, cell_statistic, components_3way, total  # noqa: E402
from implementation_facet import layer_mask  # noqa: E402

MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
PUBLISHED = {"qwen2.5-0.5b": 0.2115, "qwen2.5-1.5b": 0.2912,
             "qwen2.5-3b": 0.2924, "qwen2.5-7b": 0.1904}


def kendall_w_only(mat):
    n_m, n_t = mat.shape
    r = np.apply_along_axis(rankdata, 0, mat)
    rs = r.sum(axis=1)
    return float(12 * ((rs - rs.mean()) ** 2).sum() / (n_t ** 2 * (n_m ** 3 - n_m)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistic", default="peak", choices=["peak", "tilt"])
    ap.add_argument("--sad-pooling", default="lf", choices=["all", "filtered", "lf"])
    ap.add_argument("--n-ref", type=int, default=200, help="reference draws per template")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    loaded = {m: d for m in MODELS if (d := load(m)) is not None}
    if a.sad_pooling != "all":
        for m, d in loaded.items():
            with np.load(ROOT / "results" / "gstudy" / m / SADFILE[a.sad_pooling],
                         allow_pickle=False) as z:
                d["sad"] = z["sad"].astype(np.float64)
                d["y"] = z["sad_label"].astype(int)
    models = list(loaded)
    fam = next(iter(loaded.values()))["fam"]
    evals = sorted(k for k, v in fam.items() if v == "eval")
    deploys = sorted(k for k, v in fam.items() if v == "deploy")
    pairs = list(itertools.product(evals, deploys))

    proj = {}
    for i, m in enumerate(models):
        d = loaded[m]
        names = np.array(d["fname"])
        for j, (fe, fo) in enumerate(pairs):
            direction = unit(d["fr"][names == fe].mean(axis=0) - d["fr"][names == fo].mean(axis=0))
            proj[(i, j)] = np.einsum("ild,ld->il", d["sad"], direction)
    y = loaded[models[0]]["y"]

    stat = np.zeros((len(models), len(pairs)))
    for i in range(len(models)):
        for j in range(len(pairs)):
            stat[i, j] = cell_statistic(y, proj[(i, j)], a.statistic)
    w_obs = kendall_w_only(stat)

    # ---- reference: the world where the ranking IS a model property ----
    rng = np.random.default_rng(0)
    n_judges = len(pairs)
    ref = []
    for j in range(len(pairs)):
        judges = np.zeros((len(models), n_judges))
        for b in range(n_judges):
            idx = rng.integers(0, y.size, y.size)
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            for i in range(len(models)):
                judges[i, b] = cell_statistic(yb, proj[(i, j)][idx], a.statistic)
        ref.append(kendall_w_only(judges))
    ref = np.array(ref)

    print(f"{'=' * 92}\nIS OBSERVED CONCORDANCE CONSISTENT WITH A STABLE MODEL RANKING?"
          f"   [{a.statistic}, {a.sad_pooling}]\n{'=' * 92}")
    print(f"  observed W across {len(pairs)} real templates            {w_obs:.3f}")
    print(f"  reference W among {n_judges} item-resamples at a FIXED template:")
    print(f"     median {np.median(ref):.3f}   2.5-97.5% [{np.percentile(ref, 2.5):.3f}, "
          f"{np.percentile(ref, 97.5):.3f}]   min {ref.min():.3f}")
    below = float((ref <= w_obs).mean())
    print(f"  fraction of reference templates whose W is <= observed: {100 * below:.1f}%")
    if w_obs < np.percentile(ref, 2.5):
        print(f"\n  Observed concordance lies BELOW the reference interval. Templates agree less")
        print(f"  than item noise alone would produce if the ranking were a model property, so")
        print(f"  the observed agreement is inconsistent with a stable model ranking.")
    else:
        print(f"\n  Observed concordance lies inside the reference interval; this design cannot")
        print(f"  distinguish it from a stable ranking measured with item noise.")

    # ---- descriptive: template-averaged per-model means ----
    print(f"\n{'=' * 92}\nPER-MODEL VALUES: one template (published) vs averaged over "
          f"{len(pairs)}\n{'=' * 92}")
    print(f"  {'model':<16}{'published (k=1)':>17}{'ours, mean over templates':>28}"
          f"{'ours, min':>11}{'ours, max':>11}")
    for i, m in enumerate(models):
        print(f"  {m:<16}{PUBLISHED[m]:17.4f}{stat[i].mean():28.4f}"
              f"{stat[i].min():11.4f}{stat[i].max():11.4f}")
    pub = np.array([PUBLISHED[m] for m in models])
    print(f"\n  spread across models, published single-template : {pub.max() - pub.min():.4f}")
    print(f"  spread across models, template-averaged         : "
          f"{stat.mean(axis=1).max() - stat.mean(axis=1).min():.4f}")
    print(f"  mean spread across templates, model fixed       : "
          f"{np.mean(stat.max(axis=1) - stat.min(axis=1)):.4f}")

    # ---- sigma^2_m at each implementation ----
    print(f"\n{'=' * 92}\nBETWEEN-MODEL VARIANCE AT EACH IMPLEMENTATION\n{'=' * 92}")
    sads = {}
    for m in models:
        # Read pooled.npz from disk rather than reusing loaded[m]["sad"], which the pooling
        # swap above has already replaced -- otherwise the "alltok" level silently duplicates
        # whichever convention was selected.
        with np.load(ROOT / "results" / "gstudy" / m / "pooled.npz", allow_pickle=False) as z:
            sads[(m, "alltok")] = (z["sad"].astype(np.float64), z["sad_label"].astype(int))
    for lvl, fn in (("wsfilt", "sad_filtered.npz"), ("wsfilt_lf", "sad_filtered_lf.npz")):
        for m in models:
            p = ROOT / "results" / "gstudy" / m / fn
            if p.exists():
                with np.load(p, allow_pickle=False) as z:
                    sads[(m, lvl)] = (z["sad"].astype(np.float64), z["sad_label"].astype(int))
    levels = sorted({k[1] for k in sads})
    n_blocks = 4
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y == 0))
    blocks = [np.concatenate([p_, q]) for p_, q in
              zip(np.array_split(pos, n_blocks), np.array_split(neg, n_blocks))]
    print(f"  {'layer set':<12}{'preprocessing':<14}{'sigma^2_m':>12}{'share':>8}{'E rho^2':>10}")
    rows = {}
    for ls in ("all", "trunk", "interior"):
        for lvl in levels:
            x = np.zeros((len(models), len(pairs), n_blocks))
            for i, m in enumerate(models):
                A, yy = sads[(m, lvl)]
                names = np.array(loaded[m]["fname"])
                for j, (fe, fo) in enumerate(pairs):
                    dirn = unit(loaded[m]["fr"][names == fe].mean(axis=0)
                                - loaded[m]["fr"][names == fo].mean(axis=0))
                    pr = np.einsum("ild,ld->il", A, dirn)
                    cols = layer_mask(pr.shape[1], ls)
                    for b, idx in enumerate(blocks):
                        au = auc_cols(yy[idx], pr[np.ix_(idx, cols)])
                        x[i, j, b] = float(np.nanmax(np.abs(np.maximum(au, 1 - au) - 0.5)))
            c = components_3way(x)
            tot = total(c)
            g = c["s_m"] / (c["s_m"] + c["s_mt"] + c["s_mh"] / n_blocks
                            + c["s_mth"] / n_blocks) if c["s_m"] > 0 else 0.0
            rows[f"{ls}/{lvl}"] = {"s_m": c["s_m"], "share": c["s_m"] / tot, "g": g}
            print(f"  {ls:<12}{lvl:<14}{c['s_m']:12.2e}{100 * c['s_m'] / tot:7.1f}%{g:10.3f}")
    sms = [v["s_m"] for v in rows.values()]
    print(f"\n  sigma^2_m ranges {min(sms):.2e} to {max(sms):.2e}, a factor of "
          f"{max(sms) / max(min(sms), 1e-12):.1f} across implementations of the SAME models")
    print(f"  and the SAME items. Whether this is a real dependence or the instability of a")
    print(f"  near-boundary moment estimator on three degrees of freedom is not separable at")
    print(f"  four models; widening the model population is what would separate them.")

    out = {"statistic": a.statistic, "pooling": a.sad_pooling,
           "w_observed": w_obs, "w_reference": ref.tolist(),
           "w_reference_median": float(np.median(ref)),
           "w_reference_ci": [float(np.percentile(ref, 2.5)), float(np.percentile(ref, 97.5))],
           "frac_reference_below_observed": below,
           "per_model_mean": {m: float(stat[i].mean()) for i, m in enumerate(models)},
           "published": PUBLISHED,
           "spread_published": float(pub.max() - pub.min()),
           "spread_template_averaged": float(stat.mean(axis=1).max() - stat.mean(axis=1).min()),
           "sigma_m_by_implementation": rows}
    p = ROOT / "results" / f"concordance_equivalence_{a.statistic}_{a.sad_pooling}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
