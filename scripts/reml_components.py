"""REML variance components with model NESTED within family.

Two reasons the ANOVA moment estimators used until now have to go.

They are unstable at the boundary. Several components here are genuinely tiny, the estimator is a
difference of mean squares, and negative estimates have to be truncated at zero -- which is why
sigma^2_m moved by a factor of seven between two specifications of the same design. REML
parameterises each component as a positive quantity, so non-negativity is built in rather than
imposed afterwards, and the intervals come from the likelihood instead of a percentile bootstrap
sitting on a boundary.

They require a balanced fully-crossed design. Gemma's chat template accepts no system turn, so
the system-versus-user-only implementation level cannot be crossed with it, and any cross-family
design is therefore unbalanced. REML handles that natively.

And the structure is not what a crossed decomposition assumes. With seven models drawn from three
families, model and family are not two crossed facets: a model belongs to exactly one family. The
right decomposition is sigma^2_family plus sigma^2_model(family), and that split is what makes the
scope condition estimable rather than asserted -- a cross-family comparison and a within-family
comparison have different numerators AND different error denominators, so they need different
numbers of templates.

Model:  y = mu + family + model(family) + template + item
                + family:template + model:template + model:item + template:item + e

fitted by maximising the restricted likelihood

    -2 logL_R = log|V| + log|X'V^-1 X| + y'Py,    V = sum_i s_i Z_i Z_i' + s_e I

over log-variances, with X an intercept. Confidence intervals are profile-likelihood based: a
component's bound is where -2 logL_R rises by the chi-square(1) quantile with the others refitted.

    python scripts/reml_components.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_gtheory import auc_cols  # noqa: E402
from implementation_facet import layer_mask  # noqa: E402

V2 = ROOT / "results" / "gstudy_v2"
N_BLOCKS = 4
LAYERSET = "interior"      # pre-registered; see docs/preregistered-analysis-rules.md L1


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def load_v2(key, pooling="validated", root=None):
    """root defaults to the v2 arrays, which hold the seven models of the canonical fit. The v1.1
    arrays are a superset under the same validated pooling and are used for the cross-family
    sensitivity check in the appendix."""
    p = (root or V2) / key / "pooled.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {
            "sad": z[f"sad_{pooling}"].astype(np.float64),
            "y": z["sad_label"].astype(int),
            "fr": z[f"framing_{pooling}"].astype(np.float64),
            "fname": [str(s) for s in z["framing_name"]],
            "fam": json.loads(str(z["families"])),
            "meta": json.loads(str(z["meta"])),
        }


def design_matrices(labels):
    """labels: dict of factor -> array of level ids per observation. -> list of (name, Z)."""
    out = []
    for name, lab in labels.items():
        levels = sorted(set(lab.tolist()))
        idx = {v: i for i, v in enumerate(levels)}
        Z = np.zeros((len(lab), len(levels)))
        Z[np.arange(len(lab)), [idx[v] for v in lab]] = 1.0
        out.append((name, Z))
    return out


def neg2_reml(theta, y, X, Zs):
    s = np.exp(theta)
    n = y.size
    V = s[-1] * np.eye(n)
    for k, (_, Z) in enumerate(Zs):
        V += s[k] * (Z @ Z.T)
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e12
    logdetV = 2.0 * np.log(np.diag(L)).sum()
    Vi_X = np.linalg.solve(V, X)
    XtViX = X.T @ Vi_X
    Vi_y = np.linalg.solve(V, y)
    beta = np.linalg.solve(XtViX, X.T @ Vi_y)
    r = y - X @ beta
    quad = float(r @ np.linalg.solve(V, r))
    sign, logdetXtViX = np.linalg.slogdet(XtViX)
    return logdetV + logdetXtViX + quad


def fit_reml(y, X, Zs, fixed=None, x0=None):
    """fixed: dict index -> log-variance held constant (for profile likelihood)."""
    k = len(Zs) + 1
    free = [i for i in range(k) if not fixed or i not in fixed]
    start = x0 if x0 is not None else np.full(k, np.log(max(y.var(), 1e-8) / k))

    def obj(t_free):
        t = start.copy()
        for i, v in zip(free, t_free):
            t[i] = v
        if fixed:
            for i, v in fixed.items():
                t[i] = v
        return neg2_reml(t, y, X, Zs)

    # L-BFGS-B on the log-variances converges in tens of evaluations where Nelder-Mead needs
    # thousands, and each evaluation is a Cholesky on an n x n matrix, so the difference decides
    # whether this is minutes or hours.
    res = minimize(obj, start[free], method="L-BFGS-B",
                   options={"maxiter": 500, "maxfun": 800})
    if not np.isfinite(res.fun) or res.fun >= 1e11:
        res = minimize(obj, start[free], method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
    t = start.copy()
    for i, v in zip(free, res.x):
        t[i] = v
    if fixed:
        for i, v in fixed.items():
            t[i] = v
    return np.exp(t), res.fun, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="validated", choices=["validated", "alltokens"])
    ap.add_argument("--layerset", default=LAYERSET)
    ap.add_argument("--ci", action="store_true", help="profile-likelihood intervals (slower)")
    ap.add_argument("--data", default="v2", choices=["v2", "v11"],
                    help="v11 is a superset under the same pooling; used for the 4-family check")
    ap.add_argument("--models", default="",
                    help="comma-separated subset; the canonical fit is the seven models of the "
                         "published ladders plus gemma-2 and llama-3.2")
    # As in direction_geometry.py: a subset must name its own output file, because a subset run
    # writing to the full run's filename silently replaced a manuscript result once already.
    ap.add_argument("--suffix", default="", help="appended to the output filename; "
                                                "required when --models is used")
    a = ap.parse_args()
    if a.models and not a.suffix:
        raise SystemExit("--models without --suffix would overwrite the canonical fit; "
                         "pass --suffix to name this subset's file")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    DATA = V2 if a.data == "v2" else ROOT / "results" / "gstudy_v11"
    conf = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))
    keys = [k for k, v in conf["models"].items() if v.get("enabled")]
    if a.models:
        keys = [k for k in keys if k in a.models.split(",")]
    models = [k for k in keys if (DATA / k / "pooled.npz").exists()]
    if len(models) < 4:
        print(f"only {len(models)} models extracted; run scripts/extract_model.py --all"); return
    famof = {k: conf["models"][k]["family"] for k in models}
    print(f"{len(models)} models across {len(set(famof.values()))} families: "
          + ", ".join(f"{k}({famof[k]})" for k in models))

    # One model resident at a time. Holding all seven float64 arrays at once is several GB and
    # the 7B framing tensor alone is ~0.9GB; loading lazily keeps this comfortably in RAM.
    fam = load_v2(models[0], a.pooling, DATA)["fam"]
    evals = sorted(k for k, v in fam.items() if v == "eval")
    deploys = sorted(k for k, v in fam.items() if v == "deploy")
    pairs = list(itertools.product(evals, deploys))

    rng = np.random.default_rng(0)
    with np.load(V2 / models[0] / "pooled.npz", allow_pickle=False) as z:
        y0 = z["sad_label"].astype(int)
    pos = rng.permutation(np.flatnonzero(y0 == 1))
    neg = rng.permutation(np.flatnonzero(y0 == 0))
    blocks = [np.concatenate([p_, q]) for p_, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]

    rec = []
    for m in models:
        d = load_v2(m, a.pooling, DATA)
        names = np.array(d["fname"])
        for j, (fe, fo) in enumerate(pairs):
            dirn = unit(d["fr"][names == fe].mean(axis=0) - d["fr"][names == fo].mean(axis=0))
            pr = np.einsum("ild,ld->il", d["sad"], dirn)
            cols = layer_mask(pr.shape[1], a.layerset)
            for b, idx in enumerate(blocks):
                au = auc_cols(d["y"][idx], pr[np.ix_(idx, cols)])
                rec.append((m, famof[m], j, b,
                            float(np.nanmax(np.abs(np.maximum(au, 1 - au) - 0.5)))))
        print(f"  {m} done", flush=True)
        del d

    mm = np.array([r[0] for r in rec]); ff = np.array([r[1] for r in rec])
    tt = np.array([r[2] for r in rec]); bb = np.array([r[3] for r in rec])
    yv = np.array([r[4] for r in rec])

    labels = {
        "family": ff,
        "model(family)": mm,                                   # nested: model implies family
        "template": tt.astype(str),
        "item": bb.astype(str),
        "family:template": np.char.add(ff, np.char.add("|", tt.astype(str))),
        "model:template": np.char.add(mm, np.char.add("|", tt.astype(str))),
        "model:item": np.char.add(mm, np.char.add("|", bb.astype(str))),
        "template:item": np.char.add(tt.astype(str), np.char.add("|", bb.astype(str))),
    }
    Zs = design_matrices(labels)
    X = np.ones((yv.size, 1))
    print(f"\n  {yv.size} observations, {len(Zs)} random terms + residual", flush=True)

    s, f, theta = fit_reml(yv, X, Zs)
    names_all = [n for n, _ in Zs] + ["residual"]
    tot = s.sum()
    print(f"\n{'=' * 90}\nREML VARIANCE COMPONENTS  (model nested in family, "
          f"{a.layerset} layers, {a.pooling})\n{'=' * 90}")
    for n, v in zip(names_all, s):
        mark = "  <- at boundary, unresolved" if v / tot < 1e-4 else ""
        print(f"  {n:<20}{v:12.3e}{100 * v / tot:8.1f}%{mark}")

    comp = dict(zip(names_all, s.tolist()))
    k_t, n_i = len(pairs), N_BLOCKS

    def erho(obj_key, inter_keys):
        num = comp[obj_key]
        den = sum(comp[k] / (k_t if "template" in k else n_i) for k in inter_keys if k in comp)
        den += comp["residual"] / (k_t * n_i)
        return num / (num + den) if num + den > 0 else float("nan")

    print(f"\n{'=' * 90}\nTHE SCOPE CONDITION, ESTIMATED\n{'=' * 90}")
    g_fam1 = comp["family"] / (comp["family"] + comp["family:template"]
                               + comp["residual"] / n_i)
    g_mod1 = comp["model(family)"] / (comp["model(family)"] + comp["model:template"]
                                     + comp["model:item"] / n_i
                                     + comp["residual"] / n_i)
    print(f"  CROSS-FAMILY comparison   object = family")
    print(f"     sigma^2_family        {comp['family']:.3e}")
    print(f"     E rho^2 at k=1        {g_fam1:.3f}")
    print(f"     E rho^2 at k={k_t:<3}       {erho('family', ['family:template']):.3f}")
    print(f"  WITHIN-FAMILY comparison  object = model within family")
    print(f"     sigma^2_model(family) {comp['model(family)']:.3e}")
    print(f"     E rho^2 at k=1        {g_mod1:.3f}")
    print(f"     E rho^2 at k={k_t:<3}       "
          f"{erho('model(family)', ['model:template', 'model:item']):.3f}")

    def k_needed(obj_key, inter_key, target=0.8):
        num, s_it = comp[obj_key], comp[inter_key]
        oth = comp["residual"] / n_i
        den_allow = num * (1 - target) / target - oth
        return None if den_allow <= 0 else max(s_it / den_allow, 1.0)

    kf = k_needed("family", "family:template")
    km = k_needed("model(family)", "model:template")
    print(f"\n  templates for E rho^2 >= 0.80 at the current {n_i * 100} items:")
    print(f"     cross-family   {'no finite k' if kf is None else f'k = {kf:.0f}'}")
    print(f"     within-family  {'no finite k' if km is None else f'k = {km:.0f}'}")

    # A one-axis answer misleads here. Reporting only "no finite k" implies templates are the
    # binding constraint when the item term alone can already exceed the budget: at n item
    # blocks the residual contributes residual/n regardless of how many templates are used.
    # The informative statement is the smallest JOINT design, and which axis binds.
    def joint(obj_key, inter_t, inter_i, target=0.8):
        num = comp[obj_key]
        best = None
        for k in range(1, 2001):
            for n in range(1, 4001):
                den = (comp[inter_t] / k + (comp.get(inter_i, 0.0) / n)
                       + comp["residual"] / (k * n))
                if num / (num + den) >= target:
                    if best is None or k * n < best[0] * best[1]:
                        best = (k, n)
                    break
        return best

    print(f"\n  smallest joint design for E rho^2 >= 0.80 (k templates x n item blocks of 100):")
    joint_out = {}
    for label, obj, it, ii in (("cross-family", "family", "family:template", None),
                               ("within-family", "model(family)", "model:template",
                                "model:item")):
        b = joint(obj, it, ii) if comp[obj] > 1e-7 else None
        joint_out[label] = list(b) if b else None
        if b:
            print(f"     {label:<14} k = {b[0]:<5} n = {b[1]} blocks ({b[1] * 100} items)")
        else:
            print(f"     {label:<14} unreachable: the object variance is at the boundary, so no")
            print(f"     {'':<14} design recovers a signal that cannot be resolved at all")

    print(f"\n  CAVEAT on the family terms. sigma^2_family rests on three families with an")
    print(f"  unbalanced 4/1/2 split, so two degrees of freedom, and gemma2 contributes a single")
    print(f"  model, which partly confounds that family with that model. A boundary estimate")
    print(f"  here means UNRESOLVED at this design, not that families are alike; the family x")
    print(f"  template term at {100 * comp['family:template'] / tot:.1f}% is itself evidence that "
          f"families behave differently.")
    print(f"\n  The two differ because the numerators differ: architecture, tokenizer and training")
    print(f"  corpus vary across families and only scale varies within one. A single-template")
    print(f"  design is therefore not uniformly inadequate -- it is adequate or not depending on")
    print(f"  which comparison is being made, which is the scope condition, now measured.")

    out = {"pooling": a.pooling, "layerset": a.layerset, "n_obs": int(yv.size),
           "models": models, "families": famof, "n_templates": k_t, "n_item_blocks": n_i,
           "components": comp, "share": {k: v / tot for k, v in comp.items()},
           "neg2_reml": float(f),
           "g_cross_family_k1": float(g_fam1), "g_within_family_k1": float(g_mod1),
           "k_for_080_cross_family": kf, "k_for_080_within_family": km,
           "joint_080": joint_out,
           "boundary_components": [n for n, v in zip(names_all, s) if v / tot < 1e-4]}

    if a.ci:
        print(f"\n  profile-likelihood 95% intervals")
        crit = chi2.ppf(0.95, 1)
        for key in ("family", "model(family)", "model:template"):
            i = names_all.index(key)
            lo, hi = None, None
            for direction, store in ((-1, "lo"), (1, "hi")):
                step, cur = 0.35, theta[i]
                for _ in range(28):
                    cur += direction * step
                    _, f2, _ = fit_reml(yv, X, Zs, fixed={i: cur}, x0=theta)
                    if f2 - f > crit:
                        break
                val = float(np.exp(cur))
                if store == "lo":
                    lo = val
                else:
                    hi = val
            print(f"    {key:<18}{comp[key]:.3e}   [{lo:.3e}, {hi:.3e}]")
            out.setdefault("ci", {})[key] = [lo, hi]

    p = ROOT / "results" / f"reml_components_{a.pooling}_{a.layerset}{a.suffix}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
