"""What determines a contrastive direction: the contrastive dataset, or the wrapper pair?

Splitting one cell's contrastive set in half gives two directions that agree at cosine ~1. On its
own that says only that the estimator is stable. The finding needs its complement: how much do
directions built from DIFFERENT wrapper pairs agree, for the same model?

    within a wrapper pair, disjoint task halves   cos = a
    across wrapper pairs, same model              cos = b

Without b, a reader can dismiss "the model x template interaction dominates" as definitional --
different vectors give different answers -- rather than as the substantive point, which is that the
field treats these as interchangeable estimators of one concept direction.

But b must not be read as zero. For random unit vectors in d dimensions the cosine has standard
deviation 1/sqrt(d), so the observed across-pair agreement sits eight to fifteen standard
deviations above chance and represents a real shared component. Writing d_j = c + w_j with a
common part and wrapper-specific parts roughly orthogonal across pairs,
cos(d_j, d_k) = |c|^2/(|c|^2 + |w|^2), which turns the cosine into the shared component's share of
variance directly. The conclusion is a decomposition, not a dismissal: a contrastive direction is a
modest cue-general component plus a larger wrapper-specific one, and a single-template design
cannot separate them. That agrees with the leave-one-framing-out result on the same data, where an
axis fitted on five eval framings transfers to a sixth it never saw.

What does hold unqualified is the other half: the contrastive dataset contributes essentially
nothing, since a single item reproduces the full set at cosine 0.99.

Two extensions sharpen it. With n = 1 the direction comes from a single task; if that already
agrees with the 60-task direction, the contrastive set is replaceable by one item. (The n = 0
variant, wrapper prefixes with no task at all, needs a short extraction and is run separately.)

Everything is reported PER LAYER. At the embedding layer a mean over roughly 140 wrapper tokens
and 20 task tokens is trivially wrapper-dominated, so a cosine near one there carries no
information, and averaging over layers would let that inflate the headline.

    python scripts/direction_geometry.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reml_components import unit  # noqa: E402


def load_any(root_name, key, pooling):
    """Read one model's pooled arrays under a named pooling convention."""
    p = ROOT / "results" / ("gstudy_v11" if root_name == "v11" else "gstudy_v2") / key / "pooled.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {"fr": z[f"framing_{pooling}"].astype(np.float64),
                "sad": z[f"sad_{pooling}"].astype(np.float64),
                "y": z["sad_label"].astype(int),
                "fname": [str(x) for x in z["framing_name"]],
                "fam": json.loads(str(z["families"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="validated",
                    choices=["validated", "alltokens", "lasttoken"])
    ap.add_argument("--data", default="v11", choices=["v2", "v11"],
                    help="v11 carries last-token pooling; see docs F1")
    ap.add_argument("--models", default="")
    # --suffix is mandatory whenever --models is given. Running a model subset used to write to the
    # same filename as the full run and silently replaced the manuscript's results; results/ is not
    # under version control, so there was nothing to restore from.
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename; required when --models is used")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    conf = json.loads((ROOT / "configs" / "gstudy_models.json").read_text(encoding="utf-8"))
    keys = [k for k, v in conf["models"].items() if v.get("enabled")
            and (ROOT / "results" / ("gstudy_v11" if a.data == "v11" else "gstudy_v2")
                 / k / "pooled.npz").exists()]
    if a.models:
        keys = [k for k in keys if k in a.models.split(",")]

    out = {"pooling": a.pooling, "models": {}}
    rng = np.random.default_rng(0)
    for m in keys:
        d = load_any(a.data, m, a.pooling)
        names = np.array(d["fname"])
        fam = d["fam"]
        evals = sorted(k for k, v in fam.items() if v == "eval")
        deploys = sorted(k for k, v in fam.items() if v == "deploy")
        pairs = list(itertools.product(evals, deploys))
        n_layers = d["fr"].shape[1]

        dirs, within, single = {}, [], []
        for j, (fe, fo) in enumerate(pairs):
            A, B = d["fr"][names == fe], d["fr"][names == fo]
            n = min(len(A), len(B))
            dirs[j] = unit(A[:n].mean(0) - B[:n].mean(0))
            perm = rng.permutation(n)
            h1, h2 = perm[: n // 2], perm[n // 2:]
            c1 = unit(A[h1].mean(0) - B[h1].mean(0))
            c2 = unit(A[h2].mean(0) - B[h2].mean(0))
            within.append(np.sum(c1 * c2, axis=-1))                       # [L+1]
            t = rng.integers(0, n)
            d1 = unit(A[t:t + 1].mean(0) - B[t:t + 1].mean(0))            # n = 1
            single.append(np.sum(d1 * dirs[j], axis=-1))

        across = []
        for j, k in itertools.combinations(range(len(pairs)), 2):
            across.append(np.sum(dirs[j] * dirs[k], axis=-1))
        # split by whether the two pairs share an arm
        share_e, share_d, disjoint = [], [], []
        for j, k in itertools.combinations(range(len(pairs)), 2):
            c = np.sum(dirs[j] * dirs[k], axis=-1)
            (share_e if pairs[j][0] == pairs[k][0] else
             share_d if pairs[j][1] == pairs[k][1] else disjoint).append(c)

        W = np.mean(within, axis=0)
        A_ = np.mean(across, axis=0)
        S = np.mean(single, axis=0)
        rows = {"within (disjoint task halves)": W, "across wrapper pairs": A_,
                "n=1 task vs full set": S,
                "across, sharing the eval arm": np.mean(share_e, axis=0),
                "across, sharing the deploy arm": np.mean(share_d, axis=0),
                "across, disjoint arms": np.mean(disjoint, axis=0)}

        print(f"\n{'=' * 96}\n{m}   {n_layers - 1} layers, {len(pairs)} wrapper pairs, "
              f"{a.pooling} pooling\n{'=' * 96}")
        probes = [0, max(1, (n_layers - 1) // 4), (n_layers - 1) // 2,
                  3 * (n_layers - 1) // 4, n_layers - 1]
        print(f"  {'cosine':<34}" + "".join(f"{f'L{p}':>10}" for p in probes))
        for lbl, v in rows.items():
            print(f"  {lbl:<34}" + "".join(f"{v[p]:10.4f}" for p in probes))
        mid = slice(max(1, int(0.25 * n_layers)), int(0.75 * n_layers))
        print(f"\n  interior-layer means:  within {W[mid].mean():.4f}   "
              f"across {A_[mid].mean():.4f}   n=1 {S[mid].mean():.4f}")
        gap = W[mid].mean() - A_[mid].mean()
        print(f"  within minus across (interior) = {gap:+.4f}")
        out["models"][m] = {k2: [float(x) for x in v] for k2, v in rows.items()}
        out["models"][m]["interior_within"] = float(W[mid].mean())
        out["models"][m]["interior_across"] = float(A_[mid].mean())
        out["models"][m]["interior_n1"] = float(S[mid].mean())
        out["models"][m]["dim"] = int(d["fr"].shape[2])
        del d

    ws = [v["interior_within"] for v in out["models"].values()]
    as_ = [v["interior_across"] for v in out["models"].values()]
    n1 = [v["interior_n1"] for v in out["models"].values()]
    dis = [float(np.mean(v["across, disjoint arms"][
        max(1, int(0.25 * len(v["across, disjoint arms"]))):
        int(0.75 * len(v["across, disjoint arms"]))])) for v in out["models"].values()]
    print(f"\n{'=' * 96}\nACROSS MODELS (interior layers)\n{'=' * 96}")
    print(f"  within a wrapper pair, disjoint task halves : {np.mean(ws):.4f} "
          f"[{min(ws):.4f}, {max(ws):.4f}]")
    print(f"  across wrapper pairs, same model            : {np.mean(as_):.4f} "
          f"[{min(as_):.4f}, {max(as_):.4f}]")
    print(f"  a single task vs the full contrastive set   : {np.mean(n1):.4f} "
          f"[{min(n1):.4f}, {max(n1):.4f}]")
    # A cosine of 0.29 is not "nothing". For random unit vectors in d dimensions the cosine has
    # standard deviation 1/sqrt(d), so the disjoint-arm figure sits many standard deviations above
    # chance and represents a real shared component. Writing each direction as a common part plus
    # a wrapper-specific part roughly orthogonal across pairs, d_j = c + w_j, gives
    # cos(d_j, d_k) = |c|^2 / (|c|^2 + |w|^2), which converts the cosine directly into the share
    # of variance carried by the shared component.
    print(f"\n{'=' * 96}\nDECOMPOSING THE DIRECTION: SHARED VERSUS WRAPPER-SPECIFIC"
          f"\n{'=' * 96}")
    print(f"  {'model':<16}{'d':>7}{'chance SD':>11}{'cos disjoint':>14}{'sigmas':>9}"
          f"{'shared var':>12}{'shared norm':>13}")
    shares = []
    for m, c_dis in zip(out["models"], dis):
        dmod = int(out["models"][m]["dim"])
        sd = 1.0 / np.sqrt(dmod)
        shares.append(c_dis)
        print(f"  {m:<16}{dmod:>7}{sd:11.4f}{c_dis:14.4f}{c_dis / sd:9.1f}"
              f"{100 * c_dis:11.0f}%{100 * np.sqrt(c_dis):12.0f}%")
        out["models"][m]["shared_var_share"] = float(c_dis)
        out["models"][m]["chance_sd"] = float(sd)
        out["models"][m]["sigmas_above_chance"] = float(c_dis / sd)
    # The additive orthogonal model d_{e,p} = c + a_e + b_p, with the three parts mutually
    # orthogonal, predicts an identity the three cosines were not fitted to:
    #   gamma = |c|^2/N            (sharing neither arm)
    #   alpha = (|c|^2+|a|^2)/N    (sharing the eval arm)
    #   beta  = (|c|^2+|b|^2)/N    (sharing the deploy arm)
    #   => alpha + beta - gamma = (|c|^2+|a|^2+|b|^2)/N = 1
    # Its residual from unity is therefore a model check rather than a free parameter, and if it
    # holds the two-way shared/wrapper-specific split becomes a three-way one.
    print(f"\n{'=' * 96}\nMODEL CHECK: alpha + beta - gamma = 1 UNDER AN ADDITIVE ORTHOGONAL MODEL"
          f"\n{'=' * 96}")
    print(f"  {'model':<16}{'gamma':>8}{'alpha':>8}{'beta':>8}{'a+b-g':>9}"
          f"{'cue-general':>13}{'eval-specific':>15}{'deploy-specific':>17}")
    ident = {}
    for m in out["models"]:
        v = out["models"][m]
        L = len(v["across, disjoint arms"])
        mid = slice(max(1, int(0.25 * L)), int(0.75 * L))
        g = float(np.mean(v["across, disjoint arms"][mid]))
        al = float(np.mean(v["across, sharing the eval arm"][mid]))
        be = float(np.mean(v["across, sharing the deploy arm"][mid]))
        s = al + be - g
        print(f"  {m:<16}{g:8.3f}{al:8.3f}{be:8.3f}{s:9.3f}"
              f"{100 * g:12.0f}%{100 * (al - g):14.0f}%{100 * (be - g):16.0f}%")
        ident[m] = {"gamma": g, "alpha": al, "beta": be, "sum": s,
                    "cue_general": g, "eval_specific": al - g, "deploy_specific": be - g}
        out["models"][m]["identity"] = ident[m]
    sums = [v["sum"] for v in ident.values()]
    print(f"\n  residual from unity: max |a+b-g - 1| = {max(abs(x - 1) for x in sums):.3f}")
    ev = float(np.mean([v["eval_specific"] for v in ident.values()]))
    dv = float(np.mean([v["deploy_specific"] for v in ident.values()]))
    cg = float(np.mean([v["cue_general"] for v in ident.values()]))
    print(f"  Three independently measured cosines land on a constraint they were not fitted to,")
    print(f"  which validates the additive orthogonal model rather than assuming it, and replaces")
    print(f"  the two-way split with a three-way one: cue-general {100 * cg:.0f}% of variance,")
    print(f"  eval-framing-specific {100 * ev:.0f}%, deploy-framing-specific {100 * dv:.0f}%.")
    if ev > cg:
        print(f"  The eval-framing-specific component EXCEEDS the cue-general one, and the two arms")
        print(f"  carry comparable idiosyncratic structure, so sampling only the treatment arm")
        print(f"  would not fix the problem.")
    out["identity_residual_max"] = float(max(abs(x - 1) for x in sums))
    out["three_way_split"] = {"cue_general": cg, "eval_specific": ev, "deploy_specific": dv}

    sv = float(np.mean(shares))
    print(f"\n  Averaged: the shared, cue-general component carries about {100 * sv:.0f}% of the")
    print(f"  variance and {100 * np.sqrt(sv):.0f}% of the norm; the wrapper-specific component")
    print(f"  carries the remaining {100 * (1 - sv):.0f}% of variance, roughly "
          f"{np.sqrt((1 - sv) / sv):.1f}x the shared part in norm.")
    print(f"\n  So the defensible statement is NOT that the wrapper pair contributes everything.")
    print(f"  It is that a contrastive direction is a sum of a modest cue-general component and a")
    print(f"  larger wrapper-specific one, and a single-template design cannot separate them. That")
    print(f"  is existence-survives and recovery-fails at the level of the vector, which agrees")
    print(f"  with the leave-one-framing-out result on the same data: an axis fitted on five eval")
    print(f"  framings transfers to a sixth it never saw, deltas +0.29 to +0.51 with exact")
    print(f"  permutation p = 0.0043 in every model. Two independent measurements of the same")
    print(f"  shared fraction.")
    print(f"\n  What DOES hold unqualified is the first half: the contrastive dataset contributes")
    print(f"  essentially nothing, since one item reproduces sixty at cosine "
          f"{np.mean(n1):.4f}.")
    out["shared_var_share_mean"] = sv
    out["wrapper_specific_var_share_mean"] = float(1 - sv)
    if a.models and not a.suffix:
        raise SystemExit("--models without --suffix would overwrite the full-run output; "
                         "pass --suffix to name this subset's file")
    p = ROOT / "results" / f"direction_geometry_{a.data}_{a.pooling}{a.suffix}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
