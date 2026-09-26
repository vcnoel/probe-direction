"""Typed records shared across data, analysis, intervention, and release code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ProtocolError(RuntimeError):
    """Base class for fail-fast protocol violations."""


class ValidationError(ProtocolError):
    """Raised when an input violates a scientific or provenance invariant."""


class IncompleteRunError(ProtocolError):
    """Raised when a run lacks required cells or artifacts."""


class CacheConflictError(ProtocolError):
    """Raised when an output path belongs to a different configuration."""


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CONFIRMATION = "confirmation"


class ClaimStatus(StrEnum):
    VALIDATED = "validated"
    REPORTED_UNVERIFIED = "reported-unverified"
    FALSIFIED = "falsified"
    CONJECTURE = "conjecture"
    PLANNED = "planned"


class SpectralComparisonMode(StrEnum):
    GRAPH_ONLY = "graph-only"
    FIXED_REFERENCE_SIGNAL = "fixed-reference-signal"
    JOINT_GRAPH_AND_SIGNAL = "joint-graph-and-signal"


@dataclass(frozen=True)
class EnvironmentFactor:
    """An independently toggled environmental cue, separate from labels/outcomes."""

    name: str
    value: str | bool | int | float


@dataclass(frozen=True)
class PromptRecord:
    """One rendered prompt with all identities needed for leakage audits."""

    row_id: str
    source_task_id: str
    template_family: str
    environment_factors: tuple[EnvironmentFactor, ...]
    condition_valence: str
    content_hash: str
    rendered_text_hash: str
    task_token_hash: str
    tokenizer_revision: str
    task_span_start: int
    task_span_end: int
    split: Split
    recognition_label: str | None = None
    behavioral_outcome: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["split"] = self.split.value
        return data


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    checksum: str

    def resolved(self, base: Path | None = None) -> Path:
        value = Path(self.path)
        return value if value.is_absolute() or base is None else base / value


@dataclass(frozen=True)
class CompletenessRequirement:
    """Minimum required crossed cells for one scientific run."""

    source_task_ids: frozenset[str] = field(default_factory=frozenset)
    conditions: frozenset[str] = field(default_factory=frozenset)
    models: frozenset[str] = field(default_factory=frozenset)
    metrics: frozenset[str] = field(default_factory=frozenset)
