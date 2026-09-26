from __future__ import annotations

from dataclasses import replace

import pytest

from eval_awareness_spectral.data import (
    assign_grouped_splits,
    build_prompt_record,
    nested_group_development_folds,
    validate_split_integrity,
)
from eval_awareness_spectral.schemas import PromptRecord, Split, ValidationError


def _record(index: int, source: str, family: str, split: Split = Split.TRAIN) -> PromptRecord:
    return build_prompt_record(
        row_id=f"row-{index}",
        source_task_id=source,
        template_family=family,
        environment_factors={"cue": bool(index % 2)},
        condition_valence="evaluation" if index % 2 else "deployment",
        content=f"content {source}",
        rendered_text=f"rendered variant {index}",
        task_tokens=[index + 1],
        tokenizer_revision="tok-v1",
        task_span=(0, 1),
        split=split,
    )


def test_source_task_leakage_across_generated_variants_is_rejected() -> None:
    rows = [
        _record(0, "same-task", "family-a", Split.TRAIN),
        _record(1, "same-task", "family-b", Split.TEST),
    ]

    with pytest.raises(ValidationError, match="source_task_id leakage"):
        validate_split_integrity(rows)


def test_confirmation_template_family_leakage_is_rejected() -> None:
    rows = [
        _record(0, "task-a", "shared-family", Split.CONFIRMATION),
        _record(1, "task-b", "shared-family", Split.TRAIN),
    ]

    with pytest.raises(ValidationError, match="template-family leakage"):
        validate_split_integrity(rows)


def test_repeated_variant_is_rejected() -> None:
    row = _record(0, "task-a", "family-a")

    with pytest.raises(ValidationError, match="repeated prompt variant"):
        validate_split_integrity([row, replace(row, split=Split.TRAIN)])


def test_assignment_is_deterministic_and_keeps_variants_grouped() -> None:
    rows = [
        (
            _record(index * 2, f"task-{index}", f"family-{index}"),
            _record(index * 2 + 1, f"task-{index}", f"family-{index}"),
        )
        for index in range(8)
    ]
    flattened = [row for pair in rows for row in pair]

    first = assign_grouped_splits(flattened, seed=19)
    second = assign_grouped_splits(reversed(flattened), seed=19)
    first_by_id = {row.row_id: row.split for row in first}
    second_by_id = {row.row_id: row.split for row in second}

    assert first_by_id == second_by_id
    for index in range(len(rows)):
        assert len({row.split for row in first if row.source_task_id == f"task-{index}"}) == 1
    validate_split_integrity(first)


def test_nested_development_folds_are_group_disjoint() -> None:
    rows = [_record(index, f"task-{index}", f"family-{index}") for index in range(8)]
    folds = nested_group_development_folds(rows, outer_folds=4, inner_folds=2, seed=7)

    assert len(folds) == 4
    for fold in folds:
        for inner_validation in fold.inner_validation_source_task_ids:
            assert fold.outer_test_source_task_ids.isdisjoint(inner_validation)
