"""Implementation choices as a formal facet, crossed with templates, items and models.

Template sampling can be dismissed as a choice an author made and could have made better.
Implementation choices cannot: they are library defaults and inherited conventions, not
decisions anyone experienced as decisions, and no paper in this literature documents them. If
their variance component rivals the template's, the reported scalar is jointly determined by
three arbitrary analysis choices whose combined variance dwarfs the between-model variance.

Levels estimable from already-extracted activations:

  layer candidate set   all hidden_states (0..L) | trunk 1..L-1 (what the released code scores)
                        | interior 25-75% of depth | a single fixed relative depth
  token filter          every token | BOS and whitespace-only tokens dropped

The first matters because the statistic is a maximum: more candidates raises it by selection
alone. The second matters because the mean-over-token identity is exact only when both sides
use the same token set, and Qwen chat templates are about a tenth whitespace tokens. Levels
requiring re-extraction (chat-template variant, TransformerLens-processed activations) are
added by the same machinery once those runs exist.

DESIGN  model x template x item-block x implementation, fully crossed, one observation per
cell. The variance components come from a general n-way random-effects solve rather than a
hand-written three-way formula, so adding a facet is a configuration change.

    python scripts/implementation_facet.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gstudy_analyze import load, unit  # noqa: E402
from gstudy_gtheory import auc_cols  # noqa: E402

MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
N_BLOCKS = 4


def layer_mask(n_entries, kind):
    """Which of the L+1 hidden_states entries the maximum is taken over."""
    if kind == "all":
        return list(range(n_entries))
    if kind == "trunk":
        return list(range(1, n_entries - 1))
    if kind == "interior":
        return [i for i in range(n_entries)
                if 0.25 <= i / (n_entries - 1) <= 0.75]
    if kind == "fixed75":
        return [int(round(0.75 * (n_entries - 1)))]
    raise ValueError(kind)


def components_nway(x, names):
    """Variance components for a balanced fully-crossed random design, one obs per cell.

    For every subset S of factors the sum of squares uses the inclusion-exclusion effect array,
    and the expected mean squares satisfy

        E[MS_S] = sum over T containing S of (prod of n_f for f not in T) * sigma^2_T,

    which is solved from the largest subset downwards. Negative estimates are truncated at zero
    and reported, because a truncated zero means unresolved at this design size, not absent.
    """
    dims = dict(zip(names, x.shape))
    k = len(names)
    means = {}
    for r in range(k + 1):
        for S in combinations(range(k), r):
            axes = tuple(i for i in range(k) if i not in S)
            means[S] = x.mean(axis=axes, keepdims=True) if axes else x

    ss, df = {}, {}
    for r in range(k + 1):
        for S in combinations(range(k), r):
            eff = np.zeros_like(means[S])
            for r2 in range(len(S) + 1):
                for T in combinations(S, r2):
                    eff = eff + ((-1) ** (len(S) - len(T))) * means[T]
            outside = int(np.prod([x.shape[i] for i in range(k) if i not in S])) or 1
            ss[S] = float(outside * (eff ** 2).sum())
            df[S] = int(np.prod([x.shape[i] - 1 for i in S])) if S else 1

    sigma = {}
    for r in range(k, -1, -1):
        for S in combinations(range(k), r):
            if not S:
                continue
            ms = ss[S] / max(df[S], 1)
            acc = 0.0
            for r2 in range(len(S) + 1, k + 1):
                for T in combinations(range(k), r2):
                    if set(S) <= set(T):
                        coef = int(np.prod([x.shape[i] for i in range(k) if i not in T])) or 1
                        acc += coef * sigma[T]
            coef_s = int(np.prod([x.shape[i] for i in range(k) if i not in S])) or 1
            sigma[S] = (ms - acc) / coef_s

    raw = {"x".join(names[i] for i in S): v for S, v in sigma.items() if S}
    trunc = sorted(k_ for k_, v in raw.items() if v < 0)
    return {k_: max(v, 0.0) for k_, v in raw.items()}, trunc, dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facet", default="deploy")
    ap.add_argument("--layersets", default="all,trunk,interior,fixed75",
                    help="comma-separated. fixed75 is a single layer, so it changes the "
                         "STATISTIC (no maximum) rather than only the candidate set, and "
                         "inflates the implementation main effect for a definitional reason.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        tag = "" if a.layersets.count(",") == 3 else "_maxonly"
        a.out = str(ROOT / "results" / f"implementation_facet{tag}.json")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    loaded = {m: d for m in MODELS if (d := load(m)) is not None}
    models = list(loaded)
    fam = next(iter(loaded.values()))["fam"]
    evals = sorted(k for k, v in fam.items() if v == "eval")
    others = sorted(k for k, v in fam.items() if v == a.facet)
    pairs = list(itertools.product(evals, others))

    # Preprocessing levels on the scoring side. All three are undocumented choices, and T1
    # showed the last is large: restoring LF and removing the 1024-token truncation moved the
    # statistic by up to 0.011 and closed the reproduction gap on three of four models. The
    # CRLF level arises here from recovering prompts through a CSV round trip, but a truncation
    # length is a choice every implementation makes and almost none report.
    sad = {}
    for m in models:
        sad[(m, "alltok")] = (loaded[m]["sad"], loaded[m]["y"])
        for lvl, fn in (("wsfilt", "sad_filtered.npz"), ("wsfilt_lf", "sad_filtered_lf.npz")):
            p = ROOT / "results" / "gstudy" / m / fn
            if p.exists():
                with np.load(p, allow_pickle=False) as z:
                    sad[(m, lvl)] = (z["sad"].astype(np.float64), z["sad_label"].astype(int))
    filters = sorted({k[1] for k in sad if k[0] == models[0]})
    filters = [f for f in filters if all((m, f) in sad for m in models)]
    layersets = a.layersets.split(",")
    impls = [(ls, tf) for ls in layersets for tf in filters]
    print(f"{len(models)} models x {len(pairs)} templates x {N_BLOCKS} item-blocks "
          f"x {len(impls)} implementations "
          f"({len(layersets)} layer sets x {len(filters)} token filters)\n")

    rng = np.random.default_rng(0)
    y0 = loaded[models[0]]["y"]
    pos = rng.permutation(np.flatnonzero(y0 == 1))
    neg = rng.permutation(np.flatnonzero(y0 == 0))
    blocks = [np.concatenate([p_, q]) for p_, q in
              zip(np.array_split(pos, N_BLOCKS), np.array_split(neg, N_BLOCKS))]

    x = np.zeros((len(models), len(pairs), N_BLOCKS, len(impls)))
    for i, m in enumerate(models):
        d = loaded[m]
        names = np.array(d["fname"])
        for j, (fe, fo) in enumerate(pairs):
            direction = unit(d["fr"][names == fe].mean(axis=0) - d["fr"][names == fo].mean(axis=0))
            projs = {}
            for tf in filters:
                A, yy = sad[(m, tf)]
                projs[tf] = (np.einsum("ild,ld->il", A, direction), yy)
            for q, (ls, tf) in enumerate(impls):
                proj, yy = projs[tf]
                cols = layer_mask(proj.shape[1], ls)
                for b, idx in enumerate(blocks):
                    au = auc_cols(yy[idx], proj[np.ix_(idx, cols)])
                    x[i, j, b, q] = float(np.nanmax(np.abs(np.maximum(au, 1 - au) - 0.5)))
        print(f"  {m} done", flush=True)

    comp, trunc, dims = components_nway(x, ["model", "template", "item", "impl"])
    tot = sum(comp.values())
    print(f"\n{'=' * 88}\nVARIANCE COMPONENTS, four facets (statistic: max over the layer set of "
          f"|AUROC-0.5|)\n{'=' * 88}")
    for k_, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        flag = "  <- truncated (unresolved, not absent)" if k_ in trunc else ""
        print(f"  {k_:<34}{v:.6f}{100 * v / tot:8.1f}%{flag}")

    sm = comp["model"]
    with_model = {k_: v for k_, v in comp.items()
                  if k_.startswith("model") and k_ != "model"}
    n = {"template": len(pairs), "item": N_BLOCKS, "impl": len(impls)}
    d_rel = sum(v / np.prod([n[f] for f in k_.split("x") if f != "model"])
                for k_, v in with_model.items())
    g = sm / (sm + d_rel) if sm + d_rel > 0 else float("nan")
    print(f"\n  E rho^2 at one template, one implementation, all {N_BLOCKS} item blocks:")
    single = sum(v for k_, v in with_model.items()
                 if set(k_.split("x")) - {"model", "item"})
    d1 = single + comp.get("modelxitem", 0.0) / N_BLOCKS
    print(f"     {sm / (sm + d1):.3f}      (this is the field's design)")
    print(f"  E rho^2 averaging over ALL {len(pairs)} templates and {len(impls)} "
          f"implementations: {g:.3f}")

    print(f"\n  template component      {100 * comp['template'] / tot:5.1f}%")
    print(f"  implementation component{100 * comp['impl'] / tot:5.1f}%")
    print(f"  model component         {100 * comp['model'] / tot:5.1f}%")
    ratio = comp["impl"] / max(comp["template"], 1e-12)
    print(f"\n  implementation / template variance ratio = {ratio:.2f}")
    print(f"  Nobody selected the implementation levels; they are library defaults and inherited")
    print(f"  conventions. A component of this size therefore cannot be read as an author having")
    print(f"  chosen badly, which is the usual reply to a template-sensitivity result.")

    Path(a.out).write_text(json.dumps(
        {"components": comp, "truncated": trunc, "dims": dims,
         "n_models": len(models), "n_templates": len(pairs), "n_impls": len(impls),
         "impls": [f"{l}/{t}" for l, t in impls],
         "share": {k_: v / tot for k_, v in comp.items()},
         "g_field_design": sm / (sm + d1), "g_all_facets": g}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
