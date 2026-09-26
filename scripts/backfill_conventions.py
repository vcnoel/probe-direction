"""Stamp conventions onto results files written before result_io.py existed.

Every entry below was derived by reading the producing script, and the evidence is recorded with it
so the declaration can be checked rather than trusted. A manuscript input absent from this table is
NOT stamped: fill_numbers.py then refuses it, which is the correct outcome, because the alternative
is guessing a convention and that is the defect this whole mechanism exists to prevent.

    python scripts/backfill_conventions.py           # report only
    python scripts/backfill_conventions.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / 'scripts'))
from pipeline import GRAPH as _G  # noqa: E402
PIPELINE = {k: v[0] for k, v in _G.items()}
from result_io import (build_conventions, check_conventions,  # noqa: E402
                       check_freshness, check_producer, require_nonvacuous,
                       script_digest, source_hashes)

LF = "results/gstudy/*/sad_filtered_lf.npz"
V11 = "results/gstudy_v11/*/pooled.npz"
V2 = "results/gstudy_v2/*/pooled.npz"

# file -> (kind, fields, evidence)
TABLE = {
    "surface_baseline.json": ("measured", dict(
        pooling="lf", layers="single mid layer for the diffuseness rows; the classifier rows use no "
        "activations", model_population=["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"],  # surface_baseline.py DIFFUSE_MODELS
        scoring_array=LF, data_root="gstudy", statistic="AUROC, unfolded"),
        "surface_baseline.py loads gstudy/<m>/sad_filtered_lf.npz at L=int(0.5*(n-1))"),

    "null_control.json": ("measured", dict(
        pooling="lf", layers="interior (1..L-1)", scoring_array=LF, data_root="gstudy",
        statistic="folded max over layers"),
        "null_control.py: sad_filtered_lf.npz; peak() ranges range(1, nlay-1)"),

    "two_concept_floor.json": ("derived", dict(
        sources=["null_control.json", "refusal_decomposition_last.json"]),
        "two_concept_floor.py reads exactly those two files"),

    "corrected_decomposition_validated.json": ("measured", dict(
        pooling="validated", layers="interior (1..L-1)", scoring_array=V11, data_root="gstudy_v11",
        statistic="folded max over layers"),
        "corrected_decomposition.py: V11/<m>/pooled.npz sad_validated; cols=range(1, shape[1]-1)"),

    "direction_geometry_v11_validated.json": ("measured", dict(
        pooling="validated", layers="all, with interior means reported", scoring_array=V11,
        data_root="gstudy_v11", statistic="cosine between directions"),
        "direction_geometry.py reports per-layer cosines and interior-layer means"),

    "direction_geometry_v11_lasttoken.json": ("measured", dict(
        pooling="lasttoken", layers="all, with interior means reported", scoring_array=V11,
        data_root="gstudy_v11", statistic="cosine between directions"),
        "same script, --pooling lasttoken"),

    "direction_geometry_v11_validated_smollm2.json": ("measured", dict(
        pooling="validated", layers="all, with interior means reported", scoring_array=V11,
        data_root="gstudy_v11", statistic="cosine between directions"),
        "same script, --models smollm2-* --suffix _smollm2"),

    "slope_distribution_deploy_peak_lf.json": ("measured", dict(
        pooling="lf for scoring; all-token for the direction arrays", layers="ALL",
        scoring_array=LF, data_root="gstudy", statistic="folded max over layers"),
        "slope_distribution.py overrides d['sad'] with SADFILE['lf']; cell_statistic uses "
        "auc_cols over every column"),

    "gstudy_gtheory_deploy_peak_lf.json": ("measured", dict(
        pooling="lf for scoring; all-token for the direction arrays", layers="ALL",
        scoring_array=LF, data_root="gstudy", statistic="folded max over layers"),
        "gstudy_gtheory.py passes the full projection to cell_statistic, which maxes over "
        "auc_cols of every layer"),

    "wrapper_curves.json": ("measured", dict(
        pooling="lf for scoring; all-token for the direction arrays", layers="ALL",
        scoring_array=LF, data_root="gstudy", statistic="folded max over layers"),
        "wrapper_curves.py, aligned to gstudy_gtheory.py: cols=range(sad.shape[1])"),

    "concordance_equivalence_peak_lf.json": ("measured", dict(
        pooling="lf", layers="ALL", model_population=["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"],  # concordance_equivalence.py MODELS
        scoring_array=LF, data_root="gstudy",
        statistic="folded max over layers; Kendall W of the model ranking"),
        "same statistic path as gstudy_gtheory.py, from which it takes per-model means"),

    "arm_decomposition_trunk_peak_validated.json": ("measured", dict(
        pooling="validated", layers="trunk", scoring_array=V2, data_root="gstudy_v2",
        statistic="folded max over layers"),
        "arm_decomposition.py: gstudy_v2/pooled.npz; cols=layer_mask(..., 'trunk')"),

    "reml_components_validated_interior.json": ("measured", dict(
        pooling="validated", layers="interior", scoring_array=V2, data_root="gstudy_v2",
        statistic="folded max over layers"),
        "reml_components.py: V2 root, LAYERSET='interior'"),

    "reml_components_validated_interior_fourfam.json": ("measured", dict(
        pooling="validated", layers="interior", scoring_array=V11, data_root="gstudy_v11",
        statistic="folded max over layers"),
        "same script, --data v11 --suffix _fourfam"),

    "label_permuted_control_validated.json": ("measured", dict(
        pooling="validated", layers="interior (1..L-1)", scoring_array=V11,
        data_root="gstudy_v11", statistic="folded max over layers"),
        "label_permuted_control.py: cols=range(1, sad.shape[1]-1)"),

    "released_label_permuted.json": ("measured", dict(
        pooling="lf", layers="interior (1..L-1)", scoring_array=LF, data_root="gstudy",
        statistic="folded max over layers"),
        "released_label_permuted.py: peak() over range(1, nlay-1) on sad_filtered_lf.npz"),

    "implementation_facet.json": ("crossed", dict(
        crossed_levels=["layers: all, trunk, interior, fixed75",
                        "pooling: wsfilt, wsfilt_lf"],
        model_population=["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"],  # implementation_facet.py MODELS
        scoring_array="results/gstudy/*/{sad_filtered,sad_filtered_lf}.npz",
        data_root="gstudy", statistic="folded max over layers"),
        "implementation_facet.py crosses --layersets with the two filtered arrays; the "
        "conventions are the facet under study"),

    "counterbalance_saturation.json": ("measured", dict(
        pooling="lf", layers="interior (1..L-1)",
        scoring_array="results/field_construction/*.npz for directions; " + LF + " for scoring",
        data_root="field_construction + gstudy", statistic="cosine to the full-set direction"),
        "counterbalance_saturation.py: FC=results/field_construction"),

    "imbalance_hazard.json": ("measured", dict(
        pooling="lf", layers="interior (1..L-1)",
        scoring_array="results/field_construction/*.npz for directions; " + LF + " for scoring",
        data_root="field_construction + gstudy", statistic="folded max over layers"),
        "imbalance_hazard.py: FC directions, sad_filtered_lf scoring, range(1, nlay-1)"),

    "field_construction.json": ("measured", dict(
        pooling="answer-token, as the released implementations build directions",
        layers="interior means reported",
        scoring_array="results/field_construction/*.npz",
        data_root="field_construction", statistic="cosine between directions"),
        "field_construction.py writes the contrastive-direction cache the two scripts above read"),

    "k_requirement.json": ("derived", dict(
        sources=["gstudy_gtheory_deploy_peak.json", "gstudy_gtheory_deploy_tilt.json"]),
        "k_requirement.py reads gstudy_gtheory_deploy_<stat>.json and solves the D-study equation"),

    "reanalysis_table.json": ("derived", dict(
        sources=["reanalysis_scaling_law.json", "reanalysis_probe_depth.json",
                 "slope_distribution_deploy_peak_lf.json", "released_label_permuted.json"]),
        "reanalysis_table.py reads exactly those four"),

    "qwen_anomaly_test.json": ("derived", dict(sources=["wrapper_curves.json"]),
        "qwen_anomaly_test.py reads the 4x36 matrix only"),

    "slope_reliability.json": ("measured", dict(
        pooling="lf for scoring; all-token for the direction arrays", layers="ALL",
        scoring_array=LF, data_root="gstudy",
        statistic="folded max over layers, refit on 20 disjoint label-stratified item halves"),
        "slope_reliability.py, same array/statistic/cols as wrapper_curves.py; models from "
        "its own LADDER, recorded in the payload"),
}


def manuscript_inputs():
    src = (ROOT / "paper" / "fill_numbers.py").read_text(encoding="utf-8")
    seen, out = set(), []
    for f in re.findall(r'load\("([^"]+)"', src):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    inputs = manuscript_inputs()
    untabled, stamped, already, stale = [], [], [], []
    for name in inputs:
        p = ROOT / "results" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        # A complete conventions block is not sufficient: a derived file also needs its sources'
        # hashes, or staleness is undetectable. Re-stamp when either is missing.
        # Missing producer info is a reason to stamp, not to skip: absent is not the same as
        # matching, and check_producer is silent when the field was never recorded.
        needs = (check_conventions(d, name) + check_freshness(d, name)
                 + check_producer(d, name))
        if not needs and "producer_sha12" in (d.get("conventions") or {}):
            already.append(name)
            continue
        if name not in TABLE:
            untabled.append(name)
            continue
        # A STALE derived file must be re-run, not re-stamped. Writing current source hashes onto a
        # file computed from earlier sources would launder exactly the failure the hashes exist to
        # expose, and it would do it silently.
        if check_freshness(d, name) and not check_conventions(d, name):
            stale.append(name)
            continue
        kind, fields, evidence = TABLE[name]
        # The population is read from the payload where the analysis records one, so it cannot drift
        # from what actually ran; only files that record no model list need it declared in the table.
        if kind in ("measured", "crossed") and "model_population" not in fields:
            got = d.get("models") or list((d.get("per_model") or {}).keys())
            if not got:
                raise SystemExit(f"{name}: records no model list, so model_population must be "
                                 f"declared in TABLE above")
            fields = dict(fields, model_population=sorted(got))
        conv = build_conventions(kind, **fields)
        if kind == "derived":
            conv["source_hashes"] = source_hashes(fields["sources"])
        # The producing script is an input too. A measured file regenerated by a changed script is
        # as stale as a derived file regenerated from a changed source, and nothing was watching it.
        argv = PIPELINE.get(name)
        if argv:
            sp = ROOT / argv[0]
            if sp.exists():
                conv["producer"] = argv[0]
                # AST with docstrings stripped: comment and prose edits must not invalidate results.
                conv["producer_sha12"] = script_digest(sp)
        conv["backfilled_from"] = evidence
        if a.write:
            d["conventions"] = conv
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        stamped.append(name)

    require_nonvacuous(len(inputs), "manuscript inputs discovered", minimum=10)
    print(f"{len(inputs)} manuscript inputs")
    print(f"  already declared : {len(already)}")
    print(f"  {'stamped' if a.write else 'would stamp'} : {len(stamped)}")
    if stale:
        print(f"  STALE, must be re-run not re-stamped : {len(stale)}")
        for n in stale:
            print(f"      {n}")
    if untabled:
        print(f"  NOT IN TABLE     : {len(untabled)}")
        for n in untabled:
            print(f"      {n}")
        print("  These are not stamped. fill_numbers.py will refuse them, which is correct:")
        print("  their convention must be derived from the producing script and added above,")
        print("  or the analysis re-run through result_io.write_result.")
    if not a.write:
        print("\n  dry run; pass --write to apply")


if __name__ == "__main__":
    main()
