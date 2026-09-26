"""Leakage-safe linear probes with nested, grouped cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval_awareness_spectral.evaluation.statistics import finite_array
from eval_awareness_spectral.schemas import ValidationError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
EstimatorFactory = Callable[[], BaseEstimator]


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for every independently constructed probe pipeline."""

    pca_components: int | float | None = None
    select_k: int | str | None = None
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 2_000
    random_state: int = 0


DEFAULT_PROBE_CONFIG = ProbeConfig()


@dataclass(frozen=True)
class ProbeEvaluation:
    scores: FloatArray
    predictions: IntArray
    threshold: float
    auc: float
    estimator: BaseEstimator


@dataclass(frozen=True)
class NestedCVResult:
    scores: FloatArray
    predictions: IntArray
    thresholds: FloatArray
    auc: float


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null: FloatArray
    p_value: float


def _arrays(
    features: ArrayLike, labels: ArrayLike, groups: ArrayLike | None = None
) -> tuple[FloatArray, IntArray, NDArray[Any] | None]:
    x = finite_array(features, name="features")
    y_raw = finite_array(labels, name="labels")
    if x.ndim != 2 or y_raw.ndim != 1 or x.shape[0] != y_raw.shape[0]:
        raise ValidationError("features must be 2D and aligned with one-dimensional labels")
    unique = np.unique(y_raw)
    if unique.size != 2:
        raise ValidationError("labels must contain exactly two classes")
    y = (y_raw == unique[1]).astype(np.int64)
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and (group_array.ndim != 1 or group_array.shape != y.shape):
        raise ValidationError("groups must be one-dimensional and aligned with labels")
    return x, y, group_array


def build_probe_pipeline(config: ProbeConfig = DEFAULT_PROBE_CONFIG) -> Pipeline:
    """Construct a fresh pipeline; all learned transforms live inside it."""

    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if config.pca_components is not None:
        steps.append(("pca", PCA(n_components=config.pca_components, random_state=config.random_state)))
    if config.select_k is not None:
        steps.append(("select", SelectKBest(score_func=f_classif, k=config.select_k)))
    steps.append(
        (
            "logistic",
            LogisticRegression(max_iter=config.max_iter, random_state=config.random_state),
        )
    )
    return Pipeline(steps)


def _inner_cv(groups: NDArray[Any]) -> GroupKFold:
    count = np.unique(groups).size
    if count < 2:
        raise ValidationError("at least two training groups are required for tuning")
    return GroupKFold(n_splits=min(4, count))


def _tune(
    x: FloatArray,
    y: IntArray,
    groups: NDArray[Any],
    config: ProbeConfig,
    estimator_factory: EstimatorFactory | None,
) -> BaseEstimator:
    estimator = build_probe_pipeline(config) if estimator_factory is None else estimator_factory()
    parameters: Mapping[str, Sequence[float]] = {"logistic__C": config.c_grid}
    if not config.c_grid:
        return estimator.fit(x, y)
    search = GridSearchCV(
        estimator,
        parameters,
        scoring="roc_auc",
        cv=_inner_cv(groups),
        refit=True,
        error_score="raise",
    )
    return search.fit(x, y, groups=groups)


def _decision_scores(estimator: BaseEstimator, x: FloatArray) -> FloatArray:
    if hasattr(estimator, "decision_function"):
        result = estimator.decision_function(x)
    elif hasattr(estimator, "predict_proba"):
        result = estimator.predict_proba(x)[:, 1]
    else:
        raise ValidationError("probe estimator must expose decision_function or predict_proba")
    return np.asarray(result, dtype=np.float64)


def _training_threshold(
    x: FloatArray,
    y: IntArray,
    groups: NDArray[Any],
    estimator: BaseEstimator,
) -> float:
    """Tune a classification threshold using grouped OOF training predictions only."""

    scores = np.asarray(
        cross_val_predict(
            estimator,
            x,
            y,
            groups=groups,
            cv=_inner_cv(groups),
            method="decision_function",
        ),
        dtype=np.float64,
    )
    candidates = np.unique(scores)
    qualities = [balanced_accuracy_score(y, scores >= value) for value in candidates]
    return float(candidates[int(np.argmax(qualities))])


def nested_group_cv(
    features: ArrayLike,
    labels: ArrayLike,
    groups: ArrayLike,
    *,
    config: ProbeConfig = DEFAULT_PROBE_CONFIG,
    estimator_factory: EstimatorFactory | None = None,
) -> NestedCVResult:
    """Generate leave-group-out scores with tuning and thresholding nested per fold."""

    x, y, group_array = _arrays(features, labels, groups)
    assert group_array is not None
    if np.unique(group_array).size < 3:
        raise ValidationError("nested group CV requires at least three groups")
    scores = np.empty(y.size, dtype=np.float64)
    predictions = np.empty(y.size, dtype=np.int64)
    thresholds = np.empty(y.size, dtype=np.float64)
    for train, test in LeaveOneGroupOut().split(x, y, group_array):
        fitted = _tune(x[train], y[train], group_array[train], config, estimator_factory)
        best = fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted
        threshold = _training_threshold(x[train], y[train], group_array[train], best)
        fold_scores = _decision_scores(fitted, x[test])
        scores[test] = fold_scores
        predictions[test] = (fold_scores >= threshold).astype(np.int64)
        thresholds[test] = threshold
    return NestedCVResult(scores, predictions, thresholds, float(roc_auc_score(y, scores)))


def fit_development_evaluate_confirmation(
    development_features: ArrayLike,
    development_labels: ArrayLike,
    development_groups: ArrayLike,
    confirmation_features: ArrayLike,
    confirmation_labels: ArrayLike,
    *,
    config: ProbeConfig = DEFAULT_PROBE_CONFIG,
    estimator_factory: EstimatorFactory | None = None,
) -> ProbeEvaluation:
    """Fit and tune only on development, then perform read-only confirmation inference."""

    dev_x, dev_y, dev_groups = _arrays(
        development_features, development_labels, development_groups
    )
    assert dev_groups is not None
    confirmation_x, confirmation_y, _ = _arrays(confirmation_features, confirmation_labels)
    if dev_x.shape[1] != confirmation_x.shape[1]:
        raise ValidationError("development and confirmation feature dimensions must match")
    fitted = _tune(dev_x, dev_y, dev_groups, config, estimator_factory)
    best = fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted
    threshold = _training_threshold(dev_x, dev_y, dev_groups, best)
    scores = _decision_scores(fitted, confirmation_x)
    predictions = (scores >= threshold).astype(np.int64)
    return ProbeEvaluation(
        scores, predictions, threshold, float(roc_auc_score(confirmation_y, scores)), fitted
    )


def full_pipeline_permutation_test(
    features: ArrayLike,
    labels: ArrayLike,
    groups: ArrayLike,
    *,
    config: ProbeConfig = DEFAULT_PROBE_CONFIG,
    n_permutations: int = 1_000,
    seed: int | None = None,
    estimator_factory: EstimatorFactory | None = None,
) -> PermutationResult:
    """Rebuild, retune, and refit the complete nested pipeline for every permutation."""

    x, y, group_array = _arrays(features, labels, groups)
    assert group_array is not None
    if n_permutations < 1:
        raise ValidationError("n_permutations must be positive")
    observed_result = nested_group_cv(
        x, y, group_array, config=config, estimator_factory=estimator_factory
    )
    observed = abs(observed_result.auc - 0.5)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        # Preserve each task's label count so every grouped tuning fold remains valid.
        permuted = y.copy()
        for group in np.unique(group_array):
            selected = np.flatnonzero(group_array == group)
            permuted[selected] = rng.permutation(permuted[selected])
        result = nested_group_cv(
            x,
            permuted,
            group_array,
            config=config,
            estimator_factory=estimator_factory,
        )
        null[index] = abs(result.auc - 0.5)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))
    return PermutationResult(observed, null, p_value)
