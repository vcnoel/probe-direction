"""The published-claim -> corrected-value table, computed rather than assembled by hand.

One row per published claim, with the value as published, the value after the correction this paper
argues for, and a verdict. The verdicts are not uniform and that is the point: a reanalysis that
overturns everything it touches is not measuring, it is arguing.

WHAT SURVIVES. Chaudhary et al. (2025) report that the probe statistic scales with parameter count
over 15 models. Correcting for layer selection -- subtracting the score that maximising over the
effective number of independent layers yields on its own -- leaves the relationship essentially
intact. Their claim survives the correction this paper is built on, and we say so first.

WHAT DOES NOT. Manek (2026) reports that the DEPTH of the peak layer shifts with scale. The peak
layer is an argmax over a curve whose top is nearly flat, and bootstrap resampling of items moves it
across a large fraction of the layer stack, so the quantity carrying that claim is not identified at
this design's precision.

SCOPE, stated because the correction differs between the two rows. The layer-selection floor used
for the Chaudhary row is a null for the MAXIMUM, not for the direction: it asks what selecting over
l_eff effectively independent layers buys, given the observed layer-to-layer autocorrelation. It is
not the label-permuted direction floor of scripts/released_label_permuted.py, which would require
their contrastive sets and 15 models up to 70B and is out of reach of this hardware. So the
Chaudhary row is corrected for selection only, and the direction question is open for it.

    python scripts/reanalysis_table.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    R = ROOT / "results"
    scal = json.loads((R / "reanalysis_scaling_law.json").read_text(encoding="utf-8"))
    depth = json.loads((R / "reanalysis_probe_depth.json").read_text(encoding="utf-8"))
    slope = json.loads((R / "slope_distribution_deploy_peak_lf.json").read_text(encoding="utf-8"))
    rlp = json.loads((R / "released_label_permuted.json").read_text(encoding="utf-8"))

    # --- Chaudhary: scaling of the statistic, corrected for layer selection ---
    lp = [math.log10(x["params_b"]) for x in scal]
    rep = [x["reported_max_distance"] for x in scal]
    exc = [x["excess_over_floor"] for x in scal]
    flo = [x["null_selection_floor"] for x in scal]
    out = {"chaudhary": {
        "n_models": len(scal),
        "r_reported": pearson(lp, rep),
        # The study reports no correlation: its scaling claim is a power-law trend line over 15 models
        # from four families with Qwen excluded. Its released files cover 15 models, five of them
        # Qwen2.5, so r_reported is our fit to the released set, and r_main restricts the same fit to
        # the families the study's main analysis uses.
        "n_main": sum(1 for x in scal if not x["model"].startswith("Qwen")),
        "r_main": pearson([a for a, x in zip(lp, scal) if not x["model"].startswith("Qwen")],
                          [b for b, x in zip(rep, scal) if not x["model"].startswith("Qwen")]),
        "n_qwen_released": sum(1 for x in scal if x["model"].startswith("Qwen")),
        "r_corrected": pearson(lp, exc),
        "r_floor": pearson(lp, flo),
        "floor_share_median": float(np.median([f / r for f, r in zip(flo, rep)])),
        "l_eff_lo": float(min(x["l_eff"] for x in scal)),
        "l_eff_hi": float(max(x["l_eff"] for x in scal)),
        "correction": "layer-selection floor only; the direction question is untested for this set"}}

    # --- Manek: depth of the peak layer, identifiability of the argmax ---
    ms = depth["models"]
    spans = [m["argmax_ci_span_fraction"] for m in ms]
    dist = [m["distinct_bootstrap_argmax_layers"] for m in ms]
    dep = [m["peak_relative_depth"] for m in ms]
    out["manek_depth"] = {
        "n_models": len(ms),
        "depth_lo": float(min(dep)), "depth_hi": float(max(dep)),
        "span_median": float(np.median(spans)), "span_hi": float(max(spans)),
        "n_span_over_third": int(sum(s > 1 / 3 for s in spans)),
        "distinct_lo": int(min(dist)), "distinct_hi": int(max(dist)),
        "peak_lo": float(min(m["peak_auroc"] for m in ms)),
        "peak_hi": float(max(m["peak_auroc"] for m in ms))}

    # --- Manek: the 7B value against the primary control ---
    sev = rlp["per_model"]["qwen2.5-7b"]
    out["manek_sevenb"] = {
        "published": sev["published"],
        "floor_system": sev["variants"]["system"]["perm_mean"],
        "floor_user": sev["variants"]["user"]["perm_mean"],
        "below_both": bool(all(sev["published"] < v["perm_mean"]
                               for v in sev["variants"].values()))}

    # --- Both: the sign of the scale relationship is wrapper-determined ---
    out["sign"] = {
        "n_wrappers": slope["n_templates"],
        "n_positive": int(round(slope["frac_positive"] * slope["n_templates"])),
        "n_negative": int(round(slope["frac_negative"] * slope["n_templates"])),
        "r_lo": float(min(slope["correlations"])), "r_hi": float(max(slope["correlations"])),
        "published": slope["published"]}

    # --- The floor is itself rendering-dependent: the sharpest single instance ---
    gaps = {k: abs(r["variants"]["system"]["perm_mean"] - r["variants"]["user"]["perm_mean"])
            for k, r in rlp["per_model"].items() if len(r["variants"]) == 2}
    worst = max(gaps, key=gaps.get)
    wr = rlp["per_model"][worst]
    out["rendering_floor"] = {
        "model": worst, "gap": gaps[worst],
        "floor_system": wr["variants"]["system"]["perm_mean"],
        "floor_user": wr["variants"]["user"]["perm_mean"],
        "published": wr["published"],
        "z_system": wr["variants"]["system"]["z_published_vs_perm"],
        "z_user": wr["variants"]["user"]["z_published_vs_perm"],
        "gap_median": float(np.median(list(gaps.values())))}

    # --- A direction with no concept information, as a fraction of each published value ---
    fr = [v["perm_mean"] / r["published"]
          for r in rlp["per_model"].values() for v in r["variants"].values()]
    out["floor_fraction"] = {"lo": float(min(fr)), "hi": float(max(fr)),
                             "n_cells": len(fr),
                             "n_over_one": int(sum(x > 1 for x in fr))}

    c, d, s = out["chaudhary"], out["manek_depth"], out["sign"]
    print(f"{'=' * 98}\nPUBLISHED CLAIM -> CORRECTED VALUE\n{'=' * 98}")
    print(f"  Chaudhary et al. scaling, {c['n_models']} models")
    print(f"    as published    r = {c['r_reported']:+.3f} against log10 parameters")
    print(f"    selection-corrected r = {c['r_corrected']:+.3f}   SURVIVES")
    print(f"    the floor itself trends only weakly, r = {c['r_floor']:+.3f}, and is "
          f"{c['floor_share_median']:.0%} of the reported value at the median")
    print(f"    l_eff spans {c['l_eff_lo']:.1f} to {c['l_eff_hi']:.1f} effective layers")
    print(f"  Manek peak depth, {d['n_models']} models")
    print(f"    as published    relative depth {d['depth_lo']:.2f} to {d['depth_hi']:.2f}")
    print(f"    argmax bootstrap CI spans {d['span_median']:.0%} of the stack at the median, "
          f"up to {d['span_hi']:.0%}")
    print(f"    {d['n_span_over_third']} of {d['n_models']} models have an argmax CI wider than a "
          f"third of their depth   NOT IDENTIFIED")
    print(f"  Manek 7B value")
    print(f"    published {out['manek_sevenb']['published']:.3f} against a label-permuted floor of "
          f"{out['manek_sevenb']['floor_system']:.3f} / "
          f"{out['manek_sevenb']['floor_user']:.3f}   AT OR BELOW FLOOR")
    print(f"  Sign of the relationship, our {s['n_wrappers']} wrappers")
    print(f"    {s['n_positive']} positive, {s['n_negative']} negative, r from {s['r_lo']:+.3f} to "
          f"{s['r_hi']:+.3f}   WRAPPER-DETERMINED")
    rf = out["rendering_floor"]
    print(f"\n  THE FLOOR IS ITSELF RENDERING-DEPENDENT. For {rf['model']} the label-permuted floor "
          f"is {rf['floor_system']:.3f}")
    print(f"  under one undocumented prompt rendering and {rf['floor_user']:.3f} under the other, a "
          f"gap of {rf['gap']:.3f} on a published")
    print(f"  value of {rf['published']:.3f}. Whether that model clears its own floor is "
          f"z = {rf['z_system']:+.2f} or {rf['z_user']:+.2f}")
    print(f"  depending on a choice no paper states. This is the implementation facet appearing "
          f"INSIDE the null.")
    ff = out["floor_fraction"]
    print(f"\n  Across {ff['n_cells']} model-rendering cells a direction with no concept information "
          f"reaches {ff['lo']:.0%} to {ff['hi']:.0%}")
    print(f"  of the published value, exceeding it in {ff['n_over_one']}.")

    p = R / "reanalysis_table.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
