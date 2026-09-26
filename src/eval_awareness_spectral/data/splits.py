"""Leakage-resistant deterministic grouped data splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TypeVar

from ..schemas import PromptRecord, Split, ValidationError

_DEFAULT_PROPORTIONS: Mapping[Split, float] = {
    Split.TRAIN: 0.7,
    Split.VALIDATION: 0.1,
    Split.TEST: 0.1,
    Split.CONFIRMATION: 0.1,
}


def _stable_order(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def validate_split_integrity(records: Iterable[PromptRecord]) -> None:
    """Raise on duplicate rows, source-task leakage, or confirmation-family leakage."""

    materialized = list(records)
    row_ids: set[str] = set()
    source_splits: dict[str, set[Split]] = defaultdict(set)
    family_splits: dict[str, set[Split]] = defaultdict(set)
    for record in materialized:
        if record.row_id in row_ids:
            raise ValidationError(f"repeated prompt variant row_id: {record.row_id}")
        row_ids.add(record.row_id)
        source_splits[record.source_task_id].add(record.split)
        family_splits[record.template_family].add(record.split)
    leaked_sources = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    if leaked_sources:
        raise ValidationError(f"source_task_id leakage across splits: {leaked_sources}")
    leaked_families = sorted(
        family
        for family, splits in family_splits.items()
        if Split.CONFIRMATION in splits and len(splits) > 1
    )
    if leaked_families:
        raise ValidationError(f"template-family leakage into confirmation: {leaked_families}")


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def assign_grouped_splits(
    records: Iterable[PromptRecord],
    *,
    seed: int = 0,
    proportions: Mapping[Split, float] = _DEFAULT_PROPORTIONS,
) -> list[PromptRecord]:
    """Assign source groups deterministically with confirmation families isolated.

    Connected components of source-task IDs and template families are atomic.
    This is slightly stricter than required, but guarantees that any family
    selected for confirmation is absent from all development splits.
    """

    materialized = list(records)
    if not materialized:
        raise ValidationError("cannot split an empty dataset")
    if len({record.row_id for record in materialized}) != len(materialized):
        raise ValidationError("repeated prompt variant row_id")
    expected = set(Split)
    if set(proportions) != expected or any(value <= 0 for value in proportions.values()):
        raise ValidationError("proportions must give every split a positive weight")
    total_proportion = sum(proportions.values())
    if abs(total_proportion - 1.0) > 1e-9:
        raise ValidationError("split proportions must sum to one")

    source_nodes = {f"source:{record.source_task_id}" for record in materialized}
    family_nodes = {f"family:{record.template_family}" for record in materialized}
    groups = _DisjointSet(source_nodes | family_nodes)
    for record in materialized:
        groups.union(f"source:{record.source_task_id}", f"family:{record.template_family}")

    component_rows: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in materialized:
        component_rows[groups.find(f"source:{record.source_task_id}")].append(record)
    ordered = sorted(component_rows, key=lambda key: _stable_order(key, seed))
    targets = {split: len(materialized) * proportions[split] for split in Split}
    counts = dict.fromkeys(Split, 0)
    component_split: dict[str, Split] = {}
    for index, component in enumerate(ordered):
        # Ensure each split is represented when there are enough independent
        # components, then use the largest normalized deficit.
        if index < len(Split) and len(ordered) >= len(Split):
            chosen = tuple(Split)[index]
        else:
            chosen = max(
                Split,
                key=lambda split: (
                    (targets[split] - counts[split]) / proportions[split],
                    -tuple(Split).index(split),
                ),
            )
        component_split[component] = chosen
        counts[chosen] += len(component_rows[component])

    assigned = [
        replace(record, split=component_split[groups.find(f"source:{record.source_task_id}")])
        for record in materialized
    ]
    validate_split_integrity(assigned)
    return assigned


@dataclass(frozen=True)
class DevelopmentFold:
    """One outer held-out group fold and its nested inner validation folds."""

    outer_index: int
    outer_test_source_task_ids: frozenset[str]
    inner_validation_source_task_ids: tuple[frozenset[str], ...]


def nested_group_development_folds(
    records: Iterable[PromptRecord],
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> tuple[DevelopmentFold, ...]:
    """Build deterministic nested folds over non-confirmation source groups."""

    materialized = list(records)
    validate_split_integrity(materialized)
    development_ids = sorted(
        {
            record.source_task_id
            for record in materialized
            if record.split is not Split.CONFIRMATION
        },
        key=lambda value: _stable_order(value, seed),
    )
    if outer_folds < 2 or inner_folds < 2:
        raise ValidationError("nested folds require at least two outer and inner folds")
    if len(development_ids) < outer_folds:
        raise ValidationError("fewer development groups than outer folds")
    outer_buckets = [frozenset(development_ids[index::outer_folds]) for index in range(outer_folds)]
    result: list[DevelopmentFold] = []
    for outer_index, outer_test in enumerate(outer_buckets):
        remaining = [task_id for task_id in development_ids if task_id not in outer_test]
        if len(remaining) < inner_folds:
            raise ValidationError("fewer outer-training groups than inner folds")
        inner_order = sorted(
            remaining,
            key=lambda value: _stable_order(value, seed + outer_index + 1),
        )
        inner = tuple(frozenset(inner_order[index::inner_folds]) for index in range(inner_folds))
        result.append(
            DevelopmentFold(
                outer_index=outer_index,
                outer_test_source_task_ids=outer_test,
                inner_validation_source_task_ids=inner,
            )
        )
    return tuple(result)


_Row = TypeVar("_Row", bound=PromptRecord)


def assign_splits(records: Iterable[_Row], *, seed: int = 0) -> list[PromptRecord]:
    """Concise alias for the default grouped split protocol."""

    return assign_grouped_splits(records, seed=seed)
