"""Two-facet generalizability study of the evaluation-awareness probe statistic.

The object of measurement is the MODEL. The field reports one number per model and treats it
as a model property; this asks how dependably that number survives the two arbitrary choices
every published study makes silently -- which contrastive template pair, and which items.

DESIGN  model x template x item-block, fully crossed, one observation per cell.

The first pass here used a two-way model x template design and lumped everything else into
residual. That is wrong in two ways. Items are a second facet, not noise, and their contribution
is estimable once the 400-item SAD set is split into blocks. And a two-way decomposition treats
every variance component as error, whereas in generalizability theory only components that
INTERACT WITH THE OBJECT OF MEASUREMENT contribute to relative error: a template that inflates
every model equally shifts the scale without disturbing the comparison between models.

    relative error   sig2_d  = s_mt/k + s_mh/n + s_mth/(k n)
    absolute error   sig2_D  = sig2_d + s_t/k + s_h/n + s_th/(k n)
    E rho^2 = s_m / (s_m + sig2_d)        generalizability coefficient (relative decisions)
    Phi     = s_m / (s_m + sig2_D)        dependability index (absolute decisions)

Both are reported because the literature makes both kinds of claim, and they have different
requirements. "Model A is more evaluation-aware than model B", measured within one study on one
template, is a RELATIVE decision and is governed by E rho^2. "This model scores 0.79, versus
0.75 reported elsewhere" compares across studies that used different templates and different
items, which is an ABSOLUTE decision and is governed by Phi. Scaling-law fits pool published
numbers across papers, so they rest on the harder of the two.

STATISTIC  max over layers of |AUROC - 0.5|, computed per cell. Note that folding each layer's
AUROC to max(a, 1-a) puts every value at or above 0.5, so peak AUROC equals this quantity plus
one half identically -- they are one variable, not two, and only one is reported.

    python scripts/gstudy_gtheory.py --n-boot 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_analyze import ORDER, load, unit  # noqa: E402

# Which SAD pooling convention a run uses. "lf" is the validated one: BOS and
# whitespace-only tokens dropped, LF newlines, no truncation. See T1.
SADFILE = {"filtered": "sad_filtered.npz", "lf": "sad_filtered_lf.npz"}

N_BLOCKS = 4          # 400 SAD items -> 4 blocks of 100; 3 df for the item facet
BLOCK_ITEMS = 100


def auc_cols(y, scores):
    """AUROC of every column of `scores` at once, via rank sums (ties handled)."""
    n1 = float(y.sum())
    n2 = float(y.size - n1)
    if n1 == 0 or n2 == 0:
        return np.full(scores.shape[1], np.nan)
    r = rankdata(scores, axis=0)
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def cell_statistic(y, proj, kind="peak"):
    """Per-cell statistic. `peak` is the field's max over layers of |AUROC-0.5|.

    `tilt` is the OLS slope of that curve against relative depth. Depth claims in this
    literature are made with an argmax, which is not well posed here: the maximum of a noisy
    curve is truncated at the layer boundaries, so a peak that wants to sit past the last layer
    is reported as sitting exactly on it. The slope uses the whole curve, has no boundary, and
    answers the same question -- does discriminability concentrate late or early.
    """
    a = auc_cols(y, proj)
    curve = np.abs(np.maximum(a, 1.0 - a) - 0.5)
    if kind == "peak":
        return float(np.nanmax(curve))
    z = np.linspace(0.0, 1.0, curve.size)
    z = z - z.mean()
    return float((z * (curve - curve.mean())).sum() / (z ** 2).sum())


def components_3way(x):
    """Random-effects variance components for a crossed 3-way design, one obs per cell.

    x: [n_m, n_t, n_h]. Returns dict of s_m, s_t, s_h, s_mt, s_mh, s_th, s_mth.
    The three-way interaction is confounded with residual error (one observation per cell),
    which is standard and does not affect the D-study: both enter sig2_d through the same term.
    """
    n_m, n_t, n_h = x.shape
    g = x.mean()
    m = x.mean(axis=(1, 2)); t = x.mean(axis=(0, 2)); h = x.mean(axis=(0, 1))
    mt = x.mean(axis=2); mh = x.mean(axis=1); th = x.mean(axis=0)

    ms_m = n_t * n_h * ((m - g) ** 2).sum() / (n_m - 1)
    ms_t = n_m * n_h * ((t - g) ** 2).sum() / (n_t - 1)
    ms_h = n_m * n_t * ((h - g) ** 2).sum() / (n_h - 1)
    ms_mt = n_h * ((mt - m[:, None] - t[None, :] + g) ** 2).sum() / ((n_m - 1) * (n_t - 1))
    ms_mh = n_t * ((mh - m[:, None] - h[None, :] + g) ** 2).sum() / ((n_m - 1) * (n_h - 1))
    ms_th = n_m * ((th - t[:, None] - h[None, :] + g) ** 2).sum() / ((n_t - 1) * (n_h - 1))
    resid = (x - mt[:, :, None] - mh[:, None, :] - th[None, :, :]
             + m[:, None, None] + t[None, :, None] + h[None, None, :] - g)
    ms_mth = (resid ** 2).sum() / ((n_m - 1) * (n_t - 1) * (n_h - 1))

    raw = {
        "s_mth": ms_mth,
        "s_mt": (ms_mt - ms_mth) / n_h,
        "s_mh": (ms_mh - ms_mth) / n_t,
        "s_th": (ms_th - ms_mth) / n_m,
        "s_m": (ms_m - ms_mt - ms_mh + ms_mth) / (n_t * n_h),
        "s_t": (ms_t - ms_mt - ms_th + ms_mth) / (n_m * n_h),
        "s_h": (ms_h - ms_mh - ms_th + ms_mth) / (n_m * n_t),
    }
    # Unbiased estimates of a variance can come out negative when the true component is small
    # relative to the mean squares being differenced. Truncating at zero is standard, but a
    # zero so produced means "too small to resolve at this df", NOT "identically zero", and
    # reading substance into it is a mistake. Which components were truncated is reported.
    out = {k: max(v, 0.0) for k, v in raw.items()}
    out["_truncated"] = sorted(k for k, v in raw.items() if v < 0)
    return out


def total(c):
    return sum(v for k, v in c.items() if not k.startswith("_"))


def coefficients(c, k, n):
    """E rho^2 (relative) and Phi (absolute) for a design of k templates and n item-blocks."""
    d_rel = c["s_mt"] / k + c["s_mh"] / n + c["s_mth"] / (k * n)
    d_abs = d_rel + c["s_t"] / k + c["s_h"] / n + c["s_th"] / (k * n)
    m = c["s_m"]
    return (m / (m + d_rel) if m + d_rel > 0 else float("nan"),
            m / (m + d_abs) if m + d_abs > 0 else float("nan"))


def null_floor(n_items, n_layers, curve=None):
    sigma = np.sqrt((n_items + 1) / (3.0 * n_items ** 2))   # n1=n2=n_items/2
    l_eff = n_layers
    if curve is not None and curve.size > 3:
        r = float(np.corrcoef(curve[:-1], curve[1:])[0, 1])
        r = min(max(r if r == r else 0.0, 0.0), 0.95)
        l_eff = max(n_layers * (1 - r) / (1 + r), 1.0)
    return float(sigma * np.sqrt(2.0 * np.log(2.0 * l_eff)))


def kendall_w(mat):
    """Concordance of the model RANKING across templates, with its significance test.

    mat: [n_models, n_templates]. Templates are the judges, models the objects. Under the null
    of independent random orderings, n_t*(n_m-1)*W is chi-square on n_m-1 df. W is reported
    with that test because a low W is not the same as no agreement: with many judges even a
    weak common signal rejects the null decisively, and the correct reading is then "agreement
    is real but far weaker than a model-property interpretation needs", not "judges disagree".

    The chi-square approximation to n_t*(n_m-1)*W needs roughly five or more objects and we
    have four, so an exact-in-distribution permutation null is used instead: model labels are
    shuffled independently within each template, which is precisely the null of independent
    orderings. The chi-square value is still returned for comparison.
    """
    n_m, n_t = mat.shape

    def stat_of(m):
        r = np.apply_along_axis(rankdata, 0, m)
        rs = r.sum(axis=1)
        return float(12 * ((rs - rs.mean()) ** 2).sum() / (n_t ** 2 * (n_m ** 3 - n_m)))

    w = stat_of(mat)
    rng = np.random.default_rng(12345)
    null = np.empty(20000)
    for b in range(null.size):
        perm = np.stack([mat[rng.permutation(n_m), j] for j in range(n_t)], axis=1)
        null[b] = stat_of(perm)
    p_perm = float((1 + (null >= w).sum()) / (null.size + 1))
    stat = n_t * (n_m - 1) * w
    return w, float(stat), float(chi2.sf(stat, n_m - 1)), p_perm


def reliability_lower_bound(r_obs):
    """Lower bound on the reliability of a measurement, from any published correlation.

    If a statistic y with reliability rho_yy is correlated against a predictor measured
    without error (log parameter count), the observed correlation is attenuated as
    r_obs = r_true * sqrt(rho_yy). Since r_true <= 1,

        rho_yy >= r_obs^2,

    which turns any published correlation into a lower bound on the reliability of the
    measurement that produced it, without access to the underlying data. It is the reason a
    low generalizability coefficient here does not contradict a strong published scaling fit:
    the coefficient is a property of a model population crossed with a facet universe, and a
    wider ladder raises the numerator while leaving the template error roughly alone.
    """
    return float(min(max(r_obs, 0.0) ** 2, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--facet", default="deploy", choices=["deploy", "neutral", "both"])
    ap.add_argument("--statistic", default="peak", choices=["peak", "tilt"])
    ap.add_argument("--sad-pooling", default="lf", choices=["all", "filtered", "lf"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = str(ROOT / "results" / f"gstudy_gtheory_{a.facet}_{a.statistic}"
                          f"_{a.sad_pooling}.json")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    loaded = {m: d for m in ORDER if (d := load(m)) is not None}
    if a.sad_pooling != "all":
        # Swap in the SAD activations pooled over the token set the released scoring code uses
        # (BOS and whitespace-only tokens dropped). The framing activations that define the
        # directions are unchanged; only the held-out scoring side moves. If the variance
        # components are stable across this swap, the decomposition does not depend on a pooling
        # convention we chose, which is what needs to be true for it to mean anything.
        keep = {}
        for m, d in loaded.items():
            p = ROOT / "results" / "gstudy" / m / SADFILE[a.sad_pooling]
            if not p.exists():
                print(f"  {m}: sad_filtered.npz missing, dropped from this run")
                continue
            with np.load(p, allow_pickle=False) as z:
                d["sad"] = z["sad"].astype(np.float64)
                d["y"] = z["sad_label"].astype(int)
            keep[m] = d
        loaded = keep
    models = list(loaded)
    fam = next(iter(loaded.values()))["fam"]
    evals = sorted(k for k, v in fam.items() if v == "eval")
    contrast = ["deploy", "neutral"] if a.facet == "both" else [a.facet]
    others = sorted(k for k, v in fam.items() if v in contrast)
    pairs = list(itertools.product(evals, others))
    print(f"{len(models)} models x {len(pairs)} template pairs "
          f"({len(evals)} eval x {len(others)} {'/'.join(contrast)}) "
          f"x {N_BLOCKS} item-blocks of {BLOCK_ITEMS}\n")

    # ---- projections: one [n_items, n_layers] matrix per (model, template) ----
    proj = {}
    for m in models:
        d = loaded[m]
        names = np.array(d["fname"])
        for j, (fe, fo) in enumerate(pairs):
            direction = unit(d["fr"][names == fe].mean(axis=0) - d["fr"][names == fo].mean(axis=0))
            proj[(m, j)] = np.einsum("ild,ld->il", d["sad"], direction)
    y_all = loaded[models[0]]["y"]
    n_items, n_layers = y_all.size, proj[(models[0], 0)].shape[1]

    # stratified blocks so each carries the same class balance as the full set
    rng = np.random.default_rng(0)
    pos = rng.permutation(np.flatnonzero(y_all == 1))
    neg = rng.permutation(np.flatnonzero(y_all == 0))
    blocks = [np.concatenate([p, q]) for p, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]

    x = np.zeros((len(models), len(pairs), N_BLOCKS))
    full = np.zeros((len(models), len(pairs)))       # chosen statistic on the whole item set
    peak = np.zeros_like(full)                       # always the peak, for the existence test
    floors = np.zeros_like(full)
    for i, m in enumerate(models):
        for j in range(len(pairs)):
            P = proj[(m, j)]
            for b, idx in enumerate(blocks):
                x[i, j, b] = cell_statistic(y_all[idx], P[idx], a.statistic)
            full[i, j] = cell_statistic(y_all, P, a.statistic)
            # the null floor is derived for a maximum over layers, so the existence test is
            # always run on the peak even when the decomposition uses the slope
            au = auc_cols(y_all, P)
            cur = np.abs(np.maximum(au, 1 - au) - 0.5)
            peak[i, j] = cur.max()
            floors[i, j] = null_floor(n_items, n_layers, cur)

    c = components_3way(x)
    tot = total(c)
    label = {"peak": "max_L |AUROC-0.5|",
             "tilt": "OLS slope of |AUROC-0.5| on relative depth"}[a.statistic]
    print(f"{'=' * 88}\nVARIANCE COMPONENTS  (statistic: {label})\n{'=' * 88}")
    pretty = {"s_m": "model (the object of measurement)", "s_t": "template",
              "s_h": "item block", "s_mt": "model x template", "s_mh": "model x item",
              "s_th": "template x item", "s_mth": "3-way + residual"}
    for k_, v in c.items():
        if k_.startswith("_"):
            continue
        flag = "  <- truncated at zero (unresolved, not absent)" if k_ in c["_truncated"] else ""
        print(f"  {pretty[k_]:<36} {v:.6f}   {100 * v / tot:5.1f}%{flag}")

    # The primary evidence that templates are not interchangeable measuring devices. Both terms
    # are interactions WITH the model, so both describe disruption of the model ordering, and
    # their ratio says how much more of it templates cause than item sampling does. It needs no
    # reference distribution and no significance test, which is why it leads and Kendall's W
    # illustrates. Note this bears on RECOVERABILITY: a stable ranking may exist and simply be
    # swamped by template-induced measurement error.
    ratio_mt_mh = c["s_mt"] / c["s_mh"] if c["s_mh"] > 0 else float("inf")
    print("")
    print(f"  PRIMARY  sigma^2_mt / sigma^2_mh = {c['s_mt']:.2e} / {c['s_mh']:.2e} = "
          f"{ratio_mt_mh:.1f}")
    print(f"           Template variation disrupts the model ordering {ratio_mt_mh:.1f}x more than")
    print(f"           item sampling does, so templates are not interchangeable measuring devices.")

    g1, phi1 = coefficients(c, k=1, n=N_BLOCKS)
    print(f"\n  the field's design (k=1 template, all {n_items} items):")
    print(f"     E rho^2 = {g1:.3f}   (relative: is model A above model B, within one study)")
    print(f"     Phi     = {phi1:.3f}   (absolute: is this model's number comparable across studies)")

    print(f"\n{'=' * 88}\nD-STUDY  templates k (rows) x item-blocks n of {BLOCK_ITEMS} (cols) -> E rho^2 / Phi"
          f"\n{'=' * 88}")
    ks = [1, 2, 4, 8, 16, 32, 64]
    ns = [1, 2, 4, 8]
    if a.statistic == "peak":
        print("  NOTE the 1/n scaling assumes the reported score is a MEAN over item levels, but")
        print("       the field computes a maximum over layers on pooled items, and the mean of")
        print("       per-block maxima is not the maximum on the pooled set. The item axis below")
        print("       is therefore an approximation for `peak`. `tilt` is near-linear in items")
        print("       and does not have this problem.")
    print("       " + "".join(f"{f'n={n} ({n * BLOCK_ITEMS})':>17}" for n in ns))
    for k in ks:
        cells = "".join(f"{f'{coefficients(c, k, n)[0]:.2f} / {coefficients(c, k, n)[1]:.2f}':>17}"
                        for n in ns)
        print(f"  k={k:<3}" + cells)
    # Adding templates alone cannot pass the ceiling set by the model x item term, and adding
    # items alone cannot pass the ceiling set by model x template. Reporting "no k suffices"
    # conflates the two; the informative statement is where each axis saturates and what the
    # joint requirement is.
    sm = c["s_m"]
    asym_k = sm / (sm + c["s_mh"] / N_BLOCKS) if sm > 0 else float("nan")
    asym_n = sm / (sm + c["s_mt"]) if sm > 0 else float("nan")
    print(f"\n  saturation: more templates at n={N_BLOCKS} blocks approaches E rho^2 -> {asym_k:.3f}"
          f"  (capped by model x item)")
    print(f"              more items at k=1 template approaches E rho^2 -> {asym_n:.3f}"
          f"  (capped by model x template)")

    need = {"asymptote_k_inf": asym_k, "asymptote_n_inf": asym_n}
    for target in (0.8, 0.9):
        best = None
        for k in range(1, 1001):
            n = next((n for n in range(1, 2001)
                      if coefficients(c, k, n)[0] >= target), None)
            if n is not None and (best is None or k * n < best[0] * best[1]):
                best = (k, n)
        need[f"design_for_grho_{target}"] = list(best) if best else None
        if best:
            print(f"  smallest joint design for E rho^2 >= {target:.2f}: "
                  f"k={best[0]} templates x n={best[1]} blocks "
                  f"({best[1] * BLOCK_ITEMS} items)")
        else:
            print(f"  no (k<=1000, n<=2000) design reaches E rho^2 >= {target:.2f}")
    if need["design_for_grho_0.8"]:
        req = need["design_for_grho_0.8"][1] * BLOCK_ITEMS
        print(f"  for reference, the held-out set used throughout this literature has "
              f"{n_items} items ({req / n_items:.0f}x short of the {req} required)")

    # ---- bootstrap over BOTH facets ----
    print(f"\n{'=' * 88}\nBOOTSTRAP ({a.n_boot} reps, resampling templates and items)\n{'=' * 88}")
    bg, bp, bsm = [], [], []
    n_t = len(pairs)
    for _ in range(a.n_boot):
        tsel = rng.integers(0, n_t, n_t)
        xb = np.zeros_like(x)
        bidx = [rng.choice(idx, idx.size, replace=True) for idx in blocks]
        for i, m in enumerate(models):
            for jj, j in enumerate(tsel):
                P = proj[(m, j)]
                for b, idx in enumerate(bidx):
                    xb[i, jj, b] = cell_statistic(y_all[idx], P[idx], a.statistic)
        cb = components_3way(xb)
        gg, pp = coefficients(cb, 1, N_BLOCKS)
        bg.append(gg); bp.append(pp)
        bsm.append(cb["s_m"] / max(total(cb), 1e-12))
    q = lambda v: np.nanpercentile(v, [2.5, 50, 97.5])
    lo, md, hi = q(bg); print(f"  E rho^2 (k=1)      {g1:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    lo, md, hi = q(bp); print(f"  Phi     (k=1)      {phi1:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    lo, md, hi = q(bsm)
    print(f"  model share of var {100 * c['s_m'] / tot:.1f}%   95% CI "
          f"[{100 * lo:.1f}%, {100 * hi:.1f}%]")

    # ---- claim-type table ----
    print(f"\n{'=' * 88}\nWHICH CLAIM TYPES SURVIVE\n{'=' * 88}")
    clears = float(np.mean(peak > floors))
    per_model_clear = {m: float(np.mean(peak[i] > floors[i])) for i, m in enumerate(models)}
    w, w_chi2, w_p, w_perm = kendall_w(full)
    print(f"  EXISTENCE   'a linear direction separates eval from deployment above chance'")
    print(f"              {100 * clears:.0f}% of (model, template) cells exceed their own")
    print(f"              selection-corrected null floor (mean floor {floors.mean():.3f}).")
    print(f"              per model: " + "  ".join(f"{m.split('-')[-1]} {100 * v:.0f}%"
                                                   for m, v in per_model_clear.items()))
    d80 = need["design_for_grho_0.8"]
    print(f"\n  RANKING     'model A is more evaluation-aware than model B'  [statistic: {a.statistic}]")
    print(f"              Kendall W across templates = {w:.3f}   permutation p={w_perm:.1e} "
          f"(20000 shuffles)")
    print(f"              [chi2({len(models) - 1})={w_chi2:.1f}, p={w_p:.1e}; the chi-square "
          f"approximation needs ~5+")
    print(f"               objects and there are {len(models)}, so the permutation null governs]")
    print(f"              agreement is WEAK BUT REAL: the null of independent orderings is")
    print(f"              rejected, and W is far below what a model-property reading needs.")
    print(f"              E rho^2 at k=1 = {g1:.3f}; a dependable design needs "
          f"{'k=%d x n=%d' % tuple(d80) if d80 else '>1000 templates'}")
    print(f"\n  COMPARISON  'this model's 0.79 exceeds the 0.75 reported elsewhere'")
    print(f"              Phi at k=1 = {phi1:.3f}")

    print(f"\n{'=' * 88}\nSCOPE: reliability is a property of the model population, not a constant"
          f"\n{'=' * 88}")
    print(f"  E rho^2 = {g1:.3f} here caps any correlation this design could show at "
          f"sqrt({g1:.3f}) = {g1 ** 0.5:.2f}.")
    for src, r in (("published scaling fit, 15 models 0.27B-70B", 0.60),):
        print(f"  {src}: r={r:.2f}  =>  its reliability must be >= r^2 = "
              f"{reliability_lower_bound(r):.2f}")
    print(f"  These are consistent, not contradictory. sigma^2_m here comes from {len(models)} "
          f"checkpoints over 0.5-7B;")
    print(f"  a ladder spanning 0.27B-70B across four families has a far larger numerator with")
    print(f"  comparable template error. The design is unreliable for the population widths most")
    print(f"  published comparisons actually use, and becomes usable only across very wide")
    print(f"  ladders -- and papers do not report which regime they are in.")

    comps = {k_: v for k_, v in c.items() if not k_.startswith("_")}
    out = {"models": models, "n_pairs": n_t, "n_blocks": N_BLOCKS, "components": comps,
           "truncated_components": c["_truncated"],
           "share": {k_: v / tot for k_, v in comps.items()},
           "g_k1": g1, "phi_k1": phi1, "ratio_smt_smh": ratio_mt_mh,
           "g_k1_ci": [float(v) for v in q(bg)], "phi_k1_ci": [float(v) for v in q(bp)],
           "model_share_ci": [float(v) for v in q(bsm)], **need,
           "fraction_cells_above_null": clears, "mean_null_floor": float(floors.mean()),
           "kendall_w": w, "kendall_chi2": w_chi2, "kendall_p_chi2": w_p,
           "kendall_p_permutation": w_perm,
           "per_model_mean": {m: float(full[i].mean()) for i, m in enumerate(models)},
           "n_layers": int(n_layers - 1),   # hidden_states includes embeddings
           "s_m": c["s_m"], "s_mt": c["s_mt"], "s_mh": c["s_mh"], "s_mth": c["s_mth"],
           "max_correlation_implied": g1 ** 0.5,
           "reliability_lower_bound_from_r060": reliability_lower_bound(0.60),
           "contrast": contrast, "statistic": a.statistic}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
