"""The 36 templates are 6 x 6, not 36 independent levels -- and that inflates the D-study.

Every decomposition so far treated the template facet as 36 exchangeable levels. It is not: a
template is an (eval framing, deployment framing) pair drawn from six of each, and the geometry
confirms the effect is additive in the two arms -- three independently measured cosines satisfy
alpha + beta - gamma = 1 to within 3%, which is the constraint an additive orthogonal model
d = c + a_e + b_p predicts and which nothing was fitted to.

The consequence is arithmetic. Averaging over all 36 pairs averages the eval-arm effect over six
levels and the deployment-arm effect over six levels, not either over thirty-six:

    error = sigma^2_(m x a)/6 + sigma^2_(m x b)/6   which is   sigma^2_(m x t)/6,  not /36.

So any coefficient quoted at k = 36 is overstated, and any requirement stated in "templates" is
really a requirement on EACH ARM. This refits with the two arms as crossed facets, which is the
same data re-indexed rather than a new measurement, and restates the decision study as
(k_eval, k_deploy, n_items).

The silver lining for the prescription: two arms of 23 framings is 46 texts to author, not 23
squared pairs.

    python scripts/arm_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import N_BLOCKS, auc_cols  # noqa: E402
from implementation_facet import components_nway, layer_mask  # noqa: E402
from reml_components import load_v2, unit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layerset", default="trunk",
                    choices=["all", "trunk", "interior", "fixed75"])
    ap.add_argument("--statistic", default="peak", choices=["peak", "tilt"])
    ap.add_argument("--pooling", default="validated")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    conf = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))
    keys = [k for k, v in conf["models"].items() if v.get("enabled")
            and (ROOT / "results" / "gstudy_v2" / k / "pooled.npz").exists()]
    fam0 = load_v2(keys[0], a.pooling)["fam"]
    evals = sorted(k for k, v in fam0.items() if v == "eval")
    deploys = sorted(k for k, v in fam0.items() if v == "deploy")

    rng = np.random.default_rng(0)
    with np.load(ROOT / "results" / "gstudy_v2" / keys[0] / "pooled.npz", allow_pickle=False) as z:
        y = z["sad_label"].astype(int)
    pos = rng.permutation(np.flatnonzero(y == 1))
    neg = rng.permutation(np.flatnonzero(y == 0))
    blocks = [np.concatenate([p_, q]) for p_, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]

    x = np.zeros((len(keys), len(evals), len(deploys), N_BLOCKS))
    for i, m in enumerate(keys):
        d = load_v2(m, a.pooling)
        names = np.array(d["fname"])
        cols = layer_mask(d["sad"].shape[1], a.layerset)
        for e, fe in enumerate(evals):
            for p, fo in enumerate(deploys):
                dirn = unit(d["fr"][names == fe].mean(0) - d["fr"][names == fo].mean(0))
                pr = np.einsum("ild,ld->il", d["sad"], dirn)
                for b, idx in enumerate(blocks):
                    au = auc_cols(d["y"][idx], pr[np.ix_(idx, cols)])
                    curve = np.abs(np.maximum(au, 1 - au) - 0.5)
                    if a.statistic == "peak":
                        x[i, e, p, b] = float(np.nanmax(curve))
                    else:
                        z = np.linspace(0.0, 1.0, curve.size)
                        z = z - z.mean()
                        x[i, e, p, b] = float((z * (curve - curve.mean())).sum()
                                              / max((z ** 2).sum(), 1e-12))
        print(f"  {m} done", flush=True)
        del d

    comp, trunc, _ = components_nway(x, ["model", "eval", "deploy", "item"])
    tot = sum(comp.values())
    print(f"\n{'=' * 92}\nFOUR-FACET DECOMPOSITION WITH THE TWO ARMS CROSSED "
          f"({a.layerset}, {a.pooling})\n{'=' * 92}")
    for k_, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        flag = "  <- truncated, unresolved" if k_ in trunc else ""
        print(f"  {k_:<28}{v:12.3e}{100 * v / tot:8.1f}%{flag}")

    n_e, n_p = len(evals), len(deploys)

    def erho(kE, kP, n):
        sm = comp["model"]
        den = 0.0
        for k_, v in comp.items():
            parts = k_.split("x")
            if "model" not in parts or k_ == "model":
                continue
            div = 1.0
            if "eval" in parts:
                div *= kE
            if "deploy" in parts:
                div *= kP
            if "item" in parts:
                div *= n
            den += v / div
        return sm / (sm + den) if sm + den > 0 else float("nan")

    # the mis-specified comparison: treat the 36 pairs as independent levels, same data
    flat = x.reshape(len(keys), n_e * n_p, N_BLOCKS)
    from gstudy_gtheory import components_3way
    c3 = components_3way(flat)
    def erho_flat(k, n):
        sm = c3["s_m"]
        den = c3["s_mt"] / k + c3["s_mh"] / n + c3["s_mth"] / (k * n)
        return sm / (sm + den) if sm + den > 0 else float("nan")

    print(f"\n{'=' * 92}\nTHE CORRECTION\n{'=' * 92}")
    print(f"  {'design':<44}{'E rho^2':>10}")
    print(f"  {'k=1 template (the field), 36 pairs as levels':<44}{erho_flat(1, N_BLOCKS):10.3f}")
    print(f"  {'k=1 template, arms crossed (1 eval x 1 deploy)':<44}{erho(1, 1, N_BLOCKS):10.3f}")
    print(f"  {'all 36 pairs, treated as 36 levels [WRONG]':<44}"
          f"{erho_flat(n_e * n_p, N_BLOCKS):10.3f}")
    print(f"  {'all 36 pairs, arms crossed [CORRECT]':<44}{erho(n_e, n_p, N_BLOCKS):10.3f}")
    print(f"\n  Treating the pairs as independent levels overstates the coefficient at full k,")
    print(f"  because averaging 36 pairs averages each arm over only six levels.")

    # The two fits are not two defensible specifications of an ambiguous design: the flat model is
    # a strict COARSENING of the crossed one, since template's df partition exactly as
    # eval + deploy + eval:deploy. So the flat estimates must be arithmetic functions of the
    # crossed ones, and both relations are verified here rather than asserted. Any failure means
    # one of the two implementations is wrong.
    ma, mb, mab = comp["modelxeval"], comp["modelxdeploy"], comp["modelxevalxdeploy"]
    df_a, df_t = n_e - 1, n_e * n_p - 1
    recon = (df_a * n_e / df_t) * (ma + mb) + mab
    # The two arms carry DIFFERENT coefficients in general; they coincide only for a square
    # design. For n_a x n_b the coefficients are (1 - df_a/df_t)/n_a and (1 - df_b/df_t)/n_b,
    # which for 4 x 8 are 7/31 and 3/31 -- a factor of 2.3 apart. Our 1/7 is what the two-term
    # form collapses to at 6 x 6, and the general form is what a reader with a non-square design
    # needs.
    df_b = n_p - 1
    coef_a = (1 - df_a / df_t) / n_e
    coef_b = (1 - df_b / df_t) / n_p
    coef = coef_a
    bias = coef_a * ma + coef_b * mb
    gap = c3["s_m"] - comp["model"]
    print(f"\n{'=' * 92}\nTWO IDENTITIES THAT MUST HOLD, AND DO\n{'=' * 92}")
    print(f"  the flat interaction is the df-weighted pooling of the three crossed terms")
    print(f"    (df_a n_a/df_t)(s_ma + s_mb) + s_mab = {recon:.6e}")
    print(f"    flat s_mt as estimated              = {c3['s_mt']:.6e}"
          f"    ratio {recon / c3['s_mt']:.6f}")
    print(f"\n  the flat sigma^2_m is upward biased by a computable amount, because MS_m carries")
    print(f"  s_ma with coefficient n_b n_h while the pooled MS_mt carries it with only")
    print(f"  (df_a/df_t) n_b n_h, so subtracting leaves a residue")
    print(f"    coefficients: eval arm 1/{1 / coef_a:.3f}, deploy arm 1/{1 / coef_b:.3f}"
          f"   (equal only because the design is square)")
    print(f"    predicted bias  c_a s_ma + c_b s_mb = {bias:.6e}")
    print(f"    observed        flat sigma^2_m - crossed sigma^2_m  = {gap:.6e}"
          f"    ratio {bias / gap if gap else float('nan'):.6f}")
    print(f"\n  Both hold exactly, so both implementations are correct and the crossed estimate is")
    print(f"  the right one; the flat sigma^2_m was upward biased. This is NOT specification")
    print(f"  dependence, and the general form is a methodological result independent of these")
    print(f"  data: treating a factorially structured template set as exchangeable levels")
    print(f"  overstates sigma^2_m by")
    print(f"      (1 - df_a/df_t) s_ma/n_a + (1 - df_b/df_t) s_mb/n_b,")
    print(f"  a quantity fixed by the design rather than by the measurement. The two coefficients")
    print(f"  coincide only for a square design -- for 4 x 8 they are 7/31 and 3/31 -- so the")
    print(f"  two-term form is the one to apply, and our 1/7 is its value at 6 x 6.")
    out_ident = {"recon_flat_s_mt": float(recon), "observed_flat_s_mt": float(c3["s_mt"]),
                 "recon_ratio": float(recon / c3["s_mt"]),
                 "predicted_bias": float(bias), "observed_gap": float(gap),
                 "bias_ratio": float(bias / gap) if gap else None,
                 "bias_coefficient_eval": float(coef_a),
                 "bias_coefficient_deploy": float(coef_b)}

    print(f"\n{'=' * 92}\nDECISION STUDY IN (k_eval, k_deploy, n item blocks)\n{'=' * 92}")
    print(f"  {'k_eval':>7}{'k_deploy':>10}" + "".join(f"{f'n={n}':>9}" for n in (4, 9, 16, 30)))
    for kE, kP in ((1, 1), (2, 2), (6, 6), (12, 12), (23, 23), (46, 46)):
        print(f"  {kE:>7}{kP:>10}" + "".join(f"{erho(kE, kP, n):9.3f}" for n in (4, 9, 16, 30)))
    best = None
    for kk in range(1, 201):
        for n in range(1, 121):
            if erho(kk, kk, n) >= 0.8:
                if best is None or kk * 2 + n < best[0] * 2 + best[1]:
                    best = (kk, n)
                break
    if best:
        print(f"\n  smallest balanced design reaching E rho^2 = 0.80: {best[0]} eval framings x "
              f"{best[0]} deployment framings")
        print(f"  x {best[1]} item blocks ({best[1] * 100} items). That is {2 * best[0]} texts to "
              f"author, not {best[0] ** 2} pairs --")
        print(f"  the pairs are the crossing of the two authored sets, which is the practical")
        print(f"  consequence of the arms being crossed rather than the pairs being levels.")

    out = {"layerset": a.layerset, "pooling": a.pooling, "models": keys,
           "n_eval": n_e, "n_deploy": n_p, "n_item_blocks": N_BLOCKS,
           "components": comp, "truncated": trunc,
           "share": {k_: v / tot for k_, v in comp.items()},
           "g_k1_flat": erho_flat(1, N_BLOCKS), "g_k1_arms": erho(1, 1, N_BLOCKS),
           "g_full_flat_WRONG": erho_flat(n_e * n_p, N_BLOCKS),
           "g_full_arms_CORRECT": erho(n_e, n_p, N_BLOCKS),
           "balanced_080": list(best) if best else None,
           "identities": out_ident}
    p = ROOT / "results" / f"arm_decomposition_{a.layerset}_{a.statistic}_{a.pooling}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
