"""Statistical utilities for leakage-safe evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score

from eval_awareness_spectral.schemas import ValidationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate and percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    samples: FloatArray


def finite_array(values: ArrayLike, *, name: str = "values") -> FloatArray:
    """Return a float array, rejecting empty, non-numeric, or non-finite input."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric") from exc
    if array.size == 0:
        raise ValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite values")
    return array


def _binary_labels(labels: ArrayLike) -> NDArray[np.int64]:
    array = finite_array(labels, name="labels")
    if array.ndim != 1:
        raise ValidationError("labels must be one-dimensional")
    unique = np.unique(array)
    if unique.size != 2:
        raise ValidationError("labels must contain exactly two classes")
    return np.asarray(array == unique[1], dtype=np.int64)


def orientation_invariant_decodability(labels: ArrayLike, scores: ArrayLike) -> float:
    """Return ``max(AUC, 1 - AUC)`` so reversing a direction changes nothing."""

    y = _binary_labels(labels)
    score_array = finite_array(scores, name="scores")
    if score_array.ndim != 1 or score_array.shape != y.shape:
        raise ValidationError("scores and labels must be aligned one-dimensional arrays")
    auc = float(roc_auc_score(y, score_array))
    return max(auc, 1.0 - auc)


def above_chance_magnitude(labels: ArrayLike, scores: ArrayLike) -> float:
    """Return orientation-invariant AUC distance above chance, in ``[0, .5]``."""

    return orientation_invariant_decodability(labels, scores) - 0.5


def task_bootstrap_ci(
    values: ArrayLike,
    task_ids: ArrayLike,
    *,
    statistic: Callable[[FloatArray], float] = np.mean,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapInterval:
    """Bootstrap whole tasks, preserving all within-task observations."""

    array = finite_array(values)
    tasks = np.asarray(task_ids)
    if array.ndim != 1 or tasks.ndim != 1 or array.shape != tasks.shape:
        raise ValidationError("values and task_ids must be aligned one-dimensional arrays")
    if n_bootstrap < 1:
        raise ValidationError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence must be between zero and one")
    unique_tasks = np.unique(tasks)
    if unique_tasks.size < 2:
        raise ValidationError("at least two tasks are required for a task bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.choice(unique_tasks, size=unique_tasks.size, replace=True)
        positions = np.concatenate([np.flatnonzero(tasks == task) for task in sampled])
        draws[index] = float(statistic(array[positions]))
    if not np.all(np.isfinite(draws)):
        raise ValidationError("bootstrap statistic returned a non-finite value")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    estimate = float(statistic(array))
    if not np.isfinite(estimate):
        raise ValidationError("statistic returned a non-finite value")
    return BootstrapInterval(estimate, float(low), float(high), draws)
