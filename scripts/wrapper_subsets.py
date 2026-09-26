"""How many wrappers a wrapper-averaged statistic needs, measured by subsampling the 36.

The remedy the paper proposes is to report the statistic averaged over k wrappers. Equation 1
predicts its reliability from variance components. This measures it directly, from the per-wrapper
statistics already on disk (results/ladder_curves.json, the full-set folded maximum for every model
and wrapper), with no model fitted.

For every ladder present and every k = 1..36, draw random k-subsets of the 36 wrappers (all of them
when there are no more than N_DRAWS distinct subsets), average the statistic over the subset for
each model, and compare that model vector with the 36-wrapper average on two readings:

  sign   whether the Pearson correlation of the averaged statistic with log10 parameters has the
         sign of the same correlation computed on the 36-wrapper average (the scaling result)
  tau    Kendall's tau-b between the subset-averaged and the 36-averaged model scores (the ranking)

The reference is the 36-wrapper average on this item set, so a subset shares wrappers with its
reference and agreement rises to 1 at k = 36 by construction. The secondary reading removes that
overlap: for k <= 18, two DISJOINT k-subsets are drawn and compared with each other on the same
two readings. Thresholds are 0.95 for sign agreement and 0.80 for mean tau, with N_DRAWS draws
per k and seed 0.

The threshold k is the smallest k from which the reading stays at or above the threshold for every
larger k in its range. A reading that never gets there is stored as null.

Equation 1 is evaluated beside it from each ladder's own components (results/pod_gtheory.json) in
two forms: as printed, generalizing over wrappers and item blocks (n = 4), and with the item set
held fixed as the subsampling holds it, where model-by-item variance counts toward the universe
score rather than the error. Both give the k at which E rho^2 reaches 0.80.

    python scripts/wrapper_subsets.py
"""

from __future__ import annotations

import sys
from math import comb
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from result_io import write_result  # noqa: E402

import json  # noqa: E402

N_DRAWS = 2000
SEED = 0
SIGN_TARGET = 0.95
TAU_TARGET = 0.80
KMAX_EQ = 100000


def subsets(rng, n, k, draws):
    """All k-subsets when there are few enough, else `draws` random ones (without replacement
    inside a subset)."""
    if comb(n, k) <= draws:
        return [np.array(c) for c in combinations(range(n), k)]
    return [rng.choice(n, size=k, replace=False) for _ in range(draws)]


def readings(lp, ref, vec):
    r = float(np.corrcoef(lp, vec)[0, 1])
    t = kendalltau(vec, ref).statistic
    return r, float(t)


def first_stays(ks, vals, target):
    """Smallest k from which vals stays >= target for every larger k; None if never."""
    out = None
    for k, v in zip(reversed(ks), reversed(vals)):
        if v >= target:
            out = k
        else:
            break
    return out


def eq1_k(c, n, target, fixed_items):
    """Smallest integer k with E rho^2(k) >= target, or None if unreachable."""
    if fixed_items:
        uni = c["s_m"] + c["s_mh"] / n
        err = lambda k: c["s_mt"] / k + c["s_mth"] / (k * n)  # noqa: E731
    else:
        uni = c["s_m"]
        err = lambda k: c["s_mt"] / k + c["s_mh"] / n + c["s_mth"] / (k * n)  # noqa: E731
    e = lambda k: uni / (uni + err(k))  # noqa: E731
    if e(KMAX_EQ) < target:
        return None, e(1)
    lo, hi = 1, KMAX_EQ
    while lo < hi:
        mid = (lo + hi) // 2
        if e(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo, e(1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    lc = json.loads((ROOT / "results" / "ladder_curves.json").read_text(encoding="utf-8"))["ladders"]
    gt = json.loads((ROOT / "results" / "pod_gtheory.json").read_text(encoding="utf-8"))["ladders"]
    out = {}
    for name, d in lc.items():
        x = np.array(d["statistic"])                 # [models, wrappers]
        lp = np.log10(d["params_b"])
        nm, nw = x.shape
        ref = x.mean(axis=1)
        r_full = float(np.corrcoef(lp, ref)[0, 1])
        ks = list(range(1, nw + 1))
        sign, tau, tau_one, r_q = [], [], [], []
        for k in ks:
            rs, ts = [], []
            for s in subsets(rng, nw, k, N_DRAWS):
                r, t = readings(lp, ref, x[:, s].mean(axis=1))
                rs.append(r)
                ts.append(t)
            rs, ts = np.array(rs), np.array(ts)
            sign.append(float((np.sign(rs) == np.sign(r_full)).mean()))
            tau.append(float(np.mean(ts)))
            tau_one.append(float((ts > 1 - 1e-9).mean()))
            r_q.append([float(v) for v in np.quantile(rs, [0.05, 0.5, 0.95])])
        # disjoint pairs, k <= nw // 2
        kd = list(range(1, nw // 2 + 1))
        dsign, dtau = [], []
        for k in kd:
            agree, ts = 0, []
            for _ in range(N_DRAWS):
                perm = rng.permutation(nw)
                a, b = x[:, perm[:k]].mean(axis=1), x[:, perm[k:2 * k]].mean(axis=1)
                ra, rb = np.corrcoef(lp, a)[0, 1], np.corrcoef(lp, b)[0, 1]
                agree += int(np.sign(ra) == np.sign(rb))
                ts.append(kendalltau(a, b).statistic)
            dsign.append(agree / N_DRAWS)
            dtau.append(float(np.mean(ts)))
        rec = {
            "models": d["models"], "n_wrappers": nw, "r_full_average": r_full,
            "k": ks, "sign_agreement": sign, "mean_tau": tau, "share_tau_one": tau_one,
            "r_quantiles_05_50_95": r_q,
            "k_sign_095": first_stays(ks, sign, SIGN_TARGET),
            "k_tau_080": first_stays(ks, tau, TAU_TARGET),
            "disjoint_k": kd, "disjoint_sign_agreement": dsign, "disjoint_mean_tau": dtau,
            "disjoint_k_sign_095": first_stays(kd, dsign, SIGN_TARGET),
            "disjoint_k_tau_080": first_stays(kd, dtau, TAU_TARGET),
        }
        if name in gt:
            c, n = gt[name]["components"], gt[name]["n_blocks"]
            k_rand, e1_rand = eq1_k(c, n, 0.80, fixed_items=False)
            k_fix, e1_fix = eq1_k(c, n, 0.80, fixed_items=True)
            rec["eq1"] = {"n_blocks": n, "k_080_items_random": k_rand, "erho_k1_items_random": e1_rand,
                          "k_080_items_fixed": k_fix, "erho_k1_items_fixed": e1_fix,
                          "matches_pod_gtheory_k_for_080": k_rand == gt[name]["k_for_080"]}
        out[name] = rec
        print(f"  {name}: r_full {r_full:+.3f}  k(sign>=.95) {rec['k_sign_095']}  "
              f"k(tau>=.80) {rec['k_tau_080']}  disjoint {rec['disjoint_k_sign_095']}/"
              f"{rec['disjoint_k_tau_080']}  eq1 {rec.get('eq1', {}).get('k_080_items_random')}/"
              f"{rec.get('eq1', {}).get('k_080_items_fixed')}", flush=True)
    write_result(ROOT / "results" / "wrapper_subsets.json",
                 {"ladders": out, "n_draws": N_DRAWS, "seed": SEED,
                  "sign_target": SIGN_TARGET, "tau_target": TAU_TARGET},
                 kind="derived", sources=["ladder_curves.json", "pod_gtheory.json"],
                 statistic="folded max over layers, full scoring set, averaged over k wrappers",
                 note="additional analysis, not part of the recorded predictions")
    print("wrote results/wrapper_subsets.json")


if __name__ == "__main__":
    main()
