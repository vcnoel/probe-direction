"""Configuration loading with deterministic hashing and strict key validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import hash_mapping
from .schemas import ValidationError


@dataclass(frozen=True)
class ProtocolConfig:
    """CPU-side protocol configuration; empirical adapters consume its manifests."""

    seed: int = 42
    operator: str = "symmetric-normalized"
    symmetry_tolerance: float = 1e-10
    residual_tolerance: float = 1e-8
    confirmation_split: str = "confirmation"
    group_keys: tuple[str, ...] = ("source_task_id", "template_family")
    required_conditions: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = ()
    min_task_families_for_law: int = 3
    min_model_families_for_law: int = 3
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.operator != "symmetric-normalized":
            raise ValidationError("basis-producing operator must be 'symmetric-normalized'")
        if self.seed < 0:
            raise ValidationError("seed must be non-negative")
        if self.symmetry_tolerance <= 0 or self.residual_tolerance <= 0:
            raise ValidationError("numerical tolerances must be positive")
        allowed = {"source_task_id", "template_family", "rendered_text_hash"}
        if not self.group_keys or not set(self.group_keys) <= allowed:
            raise ValidationError(f"group_keys must be a non-empty subset of {sorted(allowed)}")
        if self.min_task_families_for_law < 2 or self.min_model_families_for_law < 2:
            raise ValidationError("law-language evidence gates must require multiple families")

    @property
    def config_hash(self) -> str:
        return hash_mapping(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "operator": self.operator,
            "symmetry_tolerance": self.symmetry_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "confirmation_split": self.confirmation_split,
            "group_keys": list(self.group_keys),
            "required_conditions": list(self.required_conditions),
            "required_metrics": list(self.required_metrics),
            "min_task_families_for_law": self.min_task_families_for_law,
            "min_model_families_for_law": self.min_model_families_for_law,
            "metadata": dict(sorted(self.metadata.items())),
        }


_FIELDS = set(ProtocolConfig.__dataclass_fields__)


def load_config(path: str | Path) -> ProtocolConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError("configuration must be a JSON object")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValidationError(f"unknown configuration keys: {sorted(unknown)}")
    for key in ("group_keys", "required_conditions", "required_metrics"):
        if key in data:
            data[key] = tuple(data[key])
    cfg = ProtocolConfig(**data)
    cfg.validate()
    return cfg
