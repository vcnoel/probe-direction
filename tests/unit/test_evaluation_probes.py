import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from eval_awareness_spectral.evaluation.probes import (
    ProbeConfig,
    build_probe_pipeline,
    fit_development_evaluate_confirmation,
    full_pipeline_permutation_test,
    nested_group_cv,
)


def _development() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), 8)
    labels = np.tile(np.repeat([0, 1], 4), 6)
    features = rng.normal(size=(48, 6))
    features[:, 0] += labels * 1.5
    return features, labels, groups


def test_nested_transforms_are_fit_only_on_each_training_fold() -> None:
    features, labels, groups = _development()
    seen_means: list[np.ndarray] = []

    class RecordingPipeline(Pipeline):
        def fit(self, x: np.ndarray, y: np.ndarray, **params: object) -> "RecordingPipeline":
            seen_means.append(np.mean(x, axis=0))
            return super().fit(x, y, **params)

    def factory() -> RecordingPipeline:
        base = build_probe_pipeline(ProbeConfig(c_grid=(1.0,)))
        return RecordingPipeline(base.steps)

    nested_group_cv(
        features,
        labels,
        groups,
        config=ProbeConfig(c_grid=(1.0,)),
        estimator_factory=factory,
    )
    full_mean = np.mean(features, axis=0)
    assert seen_means
    assert any(not np.allclose(mean, full_mean) for mean in seen_means)


def test_confirmation_never_influences_scaler_pca_or_tuning() -> None:
    features, labels, groups = _development()
    confirmation = np.arange(48, dtype=float).reshape(8, 6)
    confirmation_labels = np.tile([0, 1], 4)
    config = ProbeConfig(pca_components=3, c_grid=(0.1, 1.0))
    first = fit_development_evaluate_confirmation(
        features,
        labels,
        groups,
        confirmation,
        confirmation_labels,
        config=config,
    )
    second = fit_development_evaluate_confirmation(
        features,
        labels,
        groups,
        confirmation + 1_000_000.0,
        confirmation_labels,
        config=config,
    )
    assert isinstance(first.estimator, GridSearchCV)
    assert isinstance(second.estimator, GridSearchCV)
    first_pipe = first.estimator.best_estimator_
    second_pipe = second.estimator.best_estimator_
    assert np.array_equal(first_pipe["scale"].mean_, second_pipe["scale"].mean_)
    assert np.array_equal(first_pipe["pca"].components_, second_pipe["pca"].components_)
    assert first.estimator.best_params_ == second.estimator.best_params_
    assert first.threshold == second.threshold


def test_permutation_rebuilds_full_pipeline() -> None:
    features, labels, groups = _development()
    constructions = 0

    def factory() -> Pipeline:
        nonlocal constructions
        constructions += 1
        return build_probe_pipeline(ProbeConfig(select_k=3, c_grid=(1.0,)))

    result = full_pipeline_permutation_test(
        features,
        labels,
        groups,
        config=ProbeConfig(select_k=3, c_grid=(1.0,)),
        n_permutations=2,
        seed=7,
        estimator_factory=factory,
    )
    assert result.null.shape == (2,)
    assert constructions == 6 * 3  # one fresh estimator per outer fold and run
