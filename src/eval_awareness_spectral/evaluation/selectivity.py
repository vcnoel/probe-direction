"""Paired target-versus-negative-control selectivity estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eval_awareness_spectral.evaluation.statistics import (
    orientation_invariant_decodability,
    task_bootstrap_ci,
)
from eval_awareness_spectral.schemas import ValidationError


@dataclass(frozen=True)
class SelectivityEstimate:
    """Mean paired decodability difference and task-bootstrap uncertainty."""

    estimate: float
    low: float
    high: float
    task_ids: NDArray[np.str_]
    paired_differences: NDArray[np.float64]


def paired_decodability_selectivity(
    target_decodability: ArrayLike,
    control_decodability: ArrayLike,
    source_task_ids: ArrayLike,
    *,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> SelectivityEstimate:
    """Directly estimate the task-paired target-minus-control interaction."""

    target = np.asarray(target_decodability, dtype=np.float64)
    control = np.asarray(control_decodability, dtype=np.float64)
    tasks = np.asarray(source_task_ids)
    if target.ndim != 1 or control.ndim != 1 or tasks.ndim != 1:
        raise ValidationError("paired decodability inputs must be one-dimensional")
    if not (target.shape == control.shape == tasks.shape):
        raise ValidationError("paired decodability inputs must be aligned")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(control)):
        raise ValidationError("decodability values must be finite")
    if np.any((target < 0.5) | (target > 1.0) | (control < 0.5) | (control > 1.0)):
        raise ValidationError("orientation-invariant decodability must lie in [0.5, 1.0]")
    unique_tasks = np.unique(tasks)
    if unique_tasks.size < 2 or unique_tasks.size != tasks.size:
        raise ValidationError("provide exactly one paired decodability value per source task")
    differences = target - control
    interval = task_bootstrap_ci(
        differences,
        tasks,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return SelectivityEstimate(
        interval.estimate,
        interval.low,
        interval.high,
        tasks.astype(np.str_),
        differences,
    )


def paired_selectivity(
    labels: ArrayLike,
    target_scores: ArrayLike,
    control_scores: ArrayLike,
    source_task_ids: ArrayLike,
    *,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> SelectivityEstimate:
    """Compare orientation-invariant target/control decodability within each task.

    This is a direct paired interaction estimate. It deliberately never subtracts
    signed raw AUCs, whose arbitrary orientation can manufacture selectivity.
    """

    y = np.asarray(labels)
    target = np.asarray(target_scores)
    control = np.asarray(control_scores)
    tasks = np.asarray(source_task_ids)
    if any(value.ndim != 1 for value in (y, target, control, tasks)):
        raise ValidationError("all selectivity inputs must be one-dimensional")
    if not (y.shape == target.shape == control.shape == tasks.shape):
        raise ValidationError("all selectivity inputs must be aligned")
    unique_tasks = np.unique(tasks)
    if unique_tasks.size < 2:
        raise ValidationError("paired selectivity requires at least two source tasks")
    differences = np.empty(unique_tasks.size, dtype=np.float64)
    for index, task in enumerate(unique_tasks):
        selected = tasks == task
        try:
            target_decoding = orientation_invariant_decodability(y[selected], target[selected])
            control_decoding = orientation_invariant_decodability(y[selected], control[selected])
        except ValidationError as exc:
            raise ValidationError(f"source task {task!r} lacks both classes") from exc
        differences[index] = target_decoding - control_decoding
    interval = task_bootstrap_ci(
        differences,
        unique_tasks,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return SelectivityEstimate(
        interval.estimate,
        interval.low,
        interval.high,
        unique_tasks.astype(np.str_),
        differences,
    )


selectivity_interaction = paired_selectivity
