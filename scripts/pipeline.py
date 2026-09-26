"""The declared dependency graph for every results file the manuscript reads.

Two stale-file incidents in this project share one cause: a convention changed upstream and a
dependent was never rerun, because the graph lived in somebody's head. Source hashes make that a
detected error. A declared graph makes it an impossible one, provided the graph is declared once and
used for everything -- so this module is the single source, and both the runner and the Makefile are
generated from it.

Each entry: output file -> (command, [source result files]). Sources are results files only; the
.npz activation caches are inputs to the whole pipeline and are never rebuilt here, because they cost
GPU time and this runs on a CPU.

    python scripts/pipeline.py --list                 dependency order, no execution
    python scripts/pipeline.py --run --tier cheap     rebuild everything except the slow fits
    python scripts/pipeline.py --run                  rebuild everything
    python scripts/pipeline.py --makefile > Makefile  emit the same graph as make targets
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SMOL = "smollm2-135m,smollm2-360m,smollm2-1.7b"
TEN = ("qwen2.5-0.5b,qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,gemma2-2b,llama3.2-1b,llama3.2-3b,"
       + SMOL)

# output -> (argv after the interpreter, sources, slow?)
GRAPH: dict[str, tuple[list[str], list[str], bool]] = {
    # --- measured: read .npz only -------------------------------------------------------------
    "surface_baseline.json": (["scripts/surface_baseline.py"], [], False),
    "null_control.json": (["scripts/null_control.py"], [], False),
    # --report aggregates the cached per-model contrastive directions; without it the script
    # expects --model and fails with KeyError: None, which is how this entry was found wrong.
    "field_construction.json": (["scripts/field_construction_check.py", "--report"], [], False),
    "direction_geometry_v11_validated.json": (
        ["scripts/direction_geometry.py", "--data", "v11", "--pooling", "validated"], [], False),
    "direction_geometry_v11_lasttoken.json": (
        ["scripts/direction_geometry.py", "--data", "v11", "--pooling", "lasttoken"], [], False),
    "direction_geometry_v11_validated_smollm2.json": (
        ["scripts/direction_geometry.py", "--data", "v11", "--pooling", "validated",
         "--models", SMOL, "--suffix", "_smollm2"], [], False),
    "corrected_decomposition_validated.json": (
        ["scripts/corrected_decomposition.py", "--pooling", "validated"], [], False),
    "label_permuted_control_validated.json": (
        ["scripts/label_permuted_control.py", "--pooling", "validated"], [], False),
    "released_label_permuted.json": (
        ["scripts/released_label_permuted.py", "--report"], [], False),
    "wrapper_curves.json": (["scripts/wrapper_curves.py"], [], False),
    "slope_distribution_deploy_peak_lf.json": (
        ["scripts/slope_distribution.py", "--sad-pooling", "lf"], [], False),
    "concordance_equivalence_peak_lf.json": (
        ["scripts/concordance_equivalence.py"], [], False),
    "counterbalance_saturation.json": (["scripts/counterbalance_saturation.py"], [], False),
    "imbalance_hazard.json": (["scripts/imbalance_hazard.py"], [], False),
    "arm_decomposition_trunk_peak_validated.json": (
        ["scripts/arm_decomposition.py"], [], True),
    "implementation_facet.json": (["scripts/implementation_facet.py"], [], True),
    "gstudy_gtheory_deploy_peak_lf.json": (
        ["scripts/gstudy_gtheory.py", "--sad-pooling", "lf"], [], True),
    "reml_components_validated_interior.json": (
        ["scripts/reml_components.py"], [], True),
    "reml_components_validated_interior_fourfam.json": (
        ["scripts/reml_components.py", "--data", "v11", "--models", TEN,
         "--suffix", "_fourfam"], [], True),

    # --- derived: read other results files ----------------------------------------------------
    "two_concept_floor.json": (["scripts/two_concept_floor.py"],
                               ["null_control.json", "refusal_decomposition_last.json"], False),
    "qwen_anomaly_test.json": (["scripts/qwen_anomaly_test.py"], ["wrapper_curves.json"], False),
    # Not derived from wrapper_curves.json: it refits from the arrays on item halves, so it is a
    # measured result that happens to answer a question about the same matrix.
    "slope_reliability.json": (["scripts/slope_reliability.py"], [], True),
    "param_count_check.json": (["scripts/param_count_check.py"], ["wrapper_curves.json"], False),
    "k_requirement.json": (["scripts/k_requirement.py"],
                           ["gstudy_gtheory_deploy_peak.json",
                            "gstudy_gtheory_deploy_tilt.json"], False),
    "reanalysis_table.json": (["scripts/reanalysis_table.py"],
                              ["reanalysis_scaling_law.json", "reanalysis_probe_depth.json",
                               "slope_distribution_deploy_peak_lf.json",
                               "released_label_permuted.json"], False),
}


def order():
    """Topological order; raises on a cycle or an undeclared internal source."""
    done, out = set(), []
    known = set(GRAPH)
    while len(out) < len(GRAPH):
        progressed = False
        for name, (_, srcs, _) in GRAPH.items():
            if name in done:
                continue
            if all(s in done or s not in known for s in srcs):
                out.append(name)
                done.add(name)
                progressed = True
        if not progressed:
            raise SystemExit(f"cycle or missing source among {sorted(set(GRAPH) - done)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--makefile", action="store_true")
    ap.add_argument("--tier", default="all", choices=["all", "cheap", "slow"])
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    seq = order()

    if a.makefile:
        print("# GENERATED by scripts/pipeline.py --makefile -- do not edit; edit GRAPH there.")
        # Not sys.executable: that bakes the generating machine's interpreter into a file
        # committed to the repo, which is how an absolute local path shipped for months.
        print("PY ?= python")
        print("R  := results")
        print(".PHONY: all clean-json")
        print("all: " + " ".join(f"$(R)/{n}" for n in seq))
        for n in seq:
            argv, srcs, _ = GRAPH[n]
            dep = " ".join(f"$(R)/{s}" for s in srcs if s in GRAPH)
            print(f"\n$(R)/{n}: {dep}".rstrip())
            print("\t$(PY) " + " ".join(argv))
        print("\nclean-json:\n\trm -f $(R)/*.json")
        return

    if a.list or not a.run:
        print(f"{len(seq)} results files in dependency order "
              f"({sum(1 for n in seq if GRAPH[n][2])} slow):")
        for i, n in enumerate(seq, 1):
            argv, srcs, slow = GRAPH[n]
            tag = "SLOW" if slow else "    "
            dep = f"  <- {', '.join(srcs)}" if srcs else ""
            print(f"  {i:>2}. {tag} {n}{dep}")
        return

    todo = [n for n in seq if a.tier == "all"
            or (a.tier == "slow") == GRAPH[n][2]]
    print(f"rebuilding {len(todo)} of {len(seq)} results files (tier={a.tier})\n")
    failed = []
    for i, n in enumerate(todo, 1):
        argv = GRAPH[n][0]
        t0 = time.time()
        r = subprocess.run([PY] + argv, cwd=ROOT, capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        dt = time.time() - t0
        ok = r.returncode == 0 and (ROOT / "results" / n).exists()
        print(f"  [{i:>2}/{len(todo)}] {'ok  ' if ok else 'FAIL'} {dt:6.1f}s  {n}")
        if not ok:
            failed.append((n, (r.stderr or r.stdout).strip().splitlines()[-3:]))
    if failed:
        print("\nFAILURES:")
        for n, tail in failed:
            print(f"  {n}")
            for l in tail:
                print(f"      {l[:150]}")
        raise SystemExit(1)
    print("\nall rebuilt")


if __name__ == "__main__":
    main()
