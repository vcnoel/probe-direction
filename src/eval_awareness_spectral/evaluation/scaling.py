"""Hierarchical scaling estimates and evidence-language gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eval_awareness_spectral.evaluation.statistics import finite_array
from eval_awareness_spectral.schemas import ValidationError


@dataclass(frozen=True)
class EvidenceGate:
    """Minimum independent family counts required to use scaling-law language."""

    minimum_task_families: int = 3
    minimum_model_families: int = 3


DEFAULT_EVIDENCE_GATE = EvidenceGate()


@dataclass(frozen=True)
class HierarchicalEffectEstimate:
    """Equal-family-weighted effect with crossed cluster-bootstrap uncertainty."""

    estimate: float
    low: float
    high: float
    standard_error: float
    task_family_effects: dict[str, float]
    model_family_effects: dict[str, float]
    bootstrap_samples: NDArray[np.float64]


def _aligned_inputs(
    values: ArrayLike,
    task_ids: ArrayLike,
    task_families: ArrayLike,
    model_families: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.str_], NDArray[np.str_]]:
    response = finite_array(values)
    tasks = np.asarray(task_ids, dtype=np.str_)
    task_family = np.asarray(task_families, dtype=np.str_)
    model_family = np.asarray(model_families, dtype=np.str_)
    if response.ndim != 1 or any(
        item.ndim != 1 or item.shape != response.shape
        for item in (tasks, task_family, model_family)
    ):
        raise ValidationError("hierarchy inputs must be aligned one-dimensional arrays")
    # A task identity must not silently move between task families.
    for task in np.unique(tasks):
        if np.unique(task_family[tasks == task]).size != 1:
            raise ValidationError(f"task {task!r} belongs to multiple task families")
    return response, tasks, task_family, model_family


def _cell_matrix(
    values: NDArray[np.float64],
    task_families: NDArray[np.str_],
    model_families: NDArray[np.str_],
    task_levels: NDArray[np.str_],
    model_levels: NDArray[np.str_],
) -> NDArray[np.float64]:
    cells = np.full((task_levels.size, model_levels.size), np.nan, dtype=np.float64)
    for row, task_family in enumerate(task_levels):
        for column, model_family in enumerate(model_levels):
            selected = (task_families == task_family) & (model_families == model_family)
            if np.any(selected):
                cells[row, column] = float(np.mean(values[selected]))
    return cells


def estimate_hierarchical_effect(
    values: ArrayLike,
    task_ids: ArrayLike,
    task_families: ArrayLike,
    model_families: ArrayLike,
    *,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> HierarchicalEffectEstimate:
    """Estimate an effect while weighting task and model families equally.

    Uncertainty comes from independently resampling task-family and model-family
    clusters. The crossed bootstrap retains dependence among observations from the
    same family instead of treating rows as independent.
    """

    response, _, task_family, model_family = _aligned_inputs(
        values, task_ids, task_families, model_families
    )
    if n_bootstrap < 2:
        raise ValidationError("n_bootstrap must be at least two")
    if not 0.0 < confidence < 1.0:
        raise ValidationError("confidence must be between zero and one")
    task_levels = np.unique(task_family)
    model_levels = np.unique(model_family)
    if task_levels.size < 2 or model_levels.size < 2:
        raise ValidationError("hierarchical estimation requires at least two families per axis")
    cells = _cell_matrix(response, task_family, model_family, task_levels, model_levels)
    if np.any(np.all(np.isnan(cells), axis=0)) or np.any(np.all(np.isnan(cells), axis=1)):
        raise ValidationError("every family must have an observed hierarchy cell")
    estimate = float(np.nanmean(cells))
    task_means = np.nanmean(cells, axis=1)
    model_means = np.nanmean(cells, axis=0)
    task_effects = {
        str(level): float(value - estimate)
        for level, value in zip(task_levels, task_means, strict=True)
    }
    model_effects = {
        str(level): float(value - estimate)
        for level, value in zip(model_levels, model_means, strict=True)
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        rows = rng.integers(0, task_levels.size, size=task_levels.size)
        columns = rng.integers(0, model_levels.size, size=model_levels.size)
        sampled_cells = cells[np.ix_(rows, columns)]
        draws[index] = float(np.nanmean(sampled_cells))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return HierarchicalEffectEstimate(
        estimate,
        float(low),
        float(high),
        float(np.std(draws, ddof=1)),
        task_effects,
        model_effects,
        draws,
    )


def enforce_evidence_language(
    wording: str,
    task_families: ArrayLike,
    model_families: ArrayLike,
    *,
    gate: EvidenceGate = DEFAULT_EVIDENCE_GATE,
) -> None:
    """Refuse ``law`` claims unless both independent-family minima are met."""

    if "law" not in wording.casefold():
        return
    task_count = np.unique(np.asarray(task_families)).size
    model_count = np.unique(np.asarray(model_families)).size
    if task_count < gate.minimum_task_families or model_count < gate.minimum_model_families:
        raise ValidationError(
            "scaling-law wording requires at least "
            f"{gate.minimum_task_families} task families and "
            f"{gate.minimum_model_families} model families; got {task_count} and {model_count}"
        )


hierarchical_effect = estimate_hierarchical_effect
