"""Where the externally-released artifacts live, resolved from the environment.

Three analyses in this repo read files that are NOT ours to redistribute: the contrastive sets and
notebooks released by Manek and by Chaudhary et al., the refusal splits released by Arditi et al.,
and the SAD benchmark. Each has its own licence and its own download, so the repo records how to
reach them rather than shipping them.

Until now their locations were absolute paths on one machine, which meant every one of those scripts
failed on any other machine with a FileNotFoundError naming a directory the reader had never heard
of. Set the variable, or read an error that says what to fetch and from where.

    EAS_REANALYSIS_ROOT   clone/download root holding manek/ and arditi/
    EAS_SAD_ZIP           the SAD benchmark archive (see scripts/sad_source.py)

Nothing here is required for the manuscript to build: every number in the paper resolves from
results/*.json, which are tracked. These paths are needed only to REGENERATE those results.
"""

from __future__ import annotations

import os
from pathlib import Path

WHERE = {
    "EAS_REANALYSIS_ROOT": (
        "the released artifacts of the two reanalysed studies.\n"
        "    Expected layout:\n"
        "      $EAS_REANALYSIS_ROOT/manek/notebooks/...     (Manek 2026, released notebooks)\n"
        "      $EAS_REANALYSIS_ROOT/manek/datasets/contrastive_dataset.json\n"
        "      $EAS_REANALYSIS_ROOT/arditi/dataset/splits/  (Arditi et al. 2024, refusal splits)"),
    "EAS_SAD_ZIP": (
        "the SAD benchmark archive. Its question text is licence-restricted and is never committed\n"
        "    here; see scripts/sad_source.py for how it is read and hashed."),
}


def external(var: str) -> Path:
    """Resolve one external root, or explain precisely what is missing and why."""
    if var not in WHERE:
        raise KeyError(f"unknown external path {var!r}; expected one of {sorted(WHERE)}")
    raw = os.environ.get(var)
    if not raw:
        raise SystemExit(
            f"{var} is not set.\n"
            f"  It must point to {WHERE[var]}\n"
            "  These artifacts belong to their original authors and are not redistributed with\n"
            "  this repository. See the 'Regenerating results' section of README.md.\n"
            "  Note that the manuscript builds WITHOUT them: every printed number resolves from\n"
            "  the tracked results/*.json.")
    p = Path(raw)
    if not p.exists():
        raise SystemExit(f"{var} is set to {p}, which does not exist.")
    return p


def reanalysis(*parts) -> Path:
    """A path under the reanalysis root, e.g. reanalysis('manek', 'datasets', 'x.json')."""
    return external("EAS_REANALYSIS_ROOT").joinpath(*parts)
