from __future__ import annotations

import numpy as np
import pytest

from eval_awareness_spectral.evaluation.statistics import (
    above_chance_magnitude,
    finite_array,
    orientation_invariant_decodability,
    task_bootstrap_ci,
)
from eval_awareness_spectral.evaluation.transfer import (
    TransferIdentity,
    activation_summary_features,
    aligned_activation_features,
    validate_transfer_identities,
)
from eval_awareness_spectral.schemas import ValidationError


def test_finite_and_binary_statistic_error_paths() -> None:
    with pytest.raises(ValidationError, match="numeric"):
        finite_array([object()])
    with pytest.raises(ValidationError, match="empty"):
        finite_array([])
    with pytest.raises(ValidationError, match="finite"):
        finite_array([np.nan])
    with pytest.raises(ValidationError, match="one-dimensional"):
        orientation_invariant_decodability([[0, 1]], [0, 1])
    with pytest.raises(ValidationError, match="exactly two"):
        orientation_invariant_decodability([0, 0], [0.1, 0.2])
    with pytest.raises(ValidationError, match="aligned"):
        orientation_invariant_decodability([0, 1], [0.1])
    assert above_chance_magnitude([0, 1], [0.1, 0.9]) == pytest.approx(0.5)


def test_task_bootstrap_validation_and_nonfinite_statistic() -> None:
    with pytest.raises(ValidationError, match="aligned"):
        task_bootstrap_ci([[1.0, 2.0]], ["a", "b"])
    with pytest.raises(ValidationError, match="positive"):
        task_bootstrap_ci([1.0, 2.0], ["a", "b"], n_bootstrap=0)
    with pytest.raises(ValidationError, match="between zero and one"):
        task_bootstrap_ci([1.0, 2.0], ["a", "b"], confidence=1.0)
    with pytest.raises(ValidationError, match="at least two tasks"):
        task_bootstrap_ci([1.0, 2.0], ["a", "a"])
    with pytest.raises(ValidationError, match="non-finite"):
        task_bootstrap_ci([1.0, 2.0], ["a", "b"], n_bootstrap=2, statistic=lambda _: float("nan"))


def _identity(
    row: str,
    task: str,
    negative: str | None = None,
    family: str | None = None,
    template: str | None = None,
) -> TransferIdentity:
    return TransferIdentity(row, task, negative, family, template or f"template-{row}")


def test_transfer_identity_all_overlap_and_family_error_paths() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        validate_transfer_identities([], [_identity("b", "tb")])
    with pytest.raises(ValidationError, match="row ID"):
        validate_transfer_identities([_identity("x", "ta")], [_identity("x", "tb")])
    with pytest.raises(ValidationError, match="source task"):
        validate_transfer_identities([_identity("a", "t")], [_identity("b", "t")])
    with pytest.raises(ValidationError, match="negative-example"):
        validate_transfer_identities([_identity("a", "ta", "neg")], [_identity("b", "tb", "neg")])
    with pytest.raises(ValidationError, match="template_family"):
        validate_transfer_identities(
            [TransferIdentity("a", "ta")],
            [_identity("b", "tb", family="fb")],
        )
    with pytest.raises(ValidationError, match="requires model_family"):
        validate_transfer_identities(
            [_identity("a", "ta")],
            [_identity("b", "tb", family="fb")],
            require_held_out_model_family=True,
        )
    with pytest.raises(ValidationError, match="shares model family"):
        validate_transfer_identities(
            [_identity("a", "ta", family="f")],
            [_identity("b", "tb", family="f")],
            require_held_out_model_family=True,
        )
    with pytest.raises(ValidationError, match="template family"):
        validate_transfer_identities(
            [_identity("a", "ta", template="shared")],
            [_identity("b", "tb", template="shared")],
        )
    validate_transfer_identities(
        [_identity("a", "ta", "na", "fa")],
        [_identity("b", "tb", "nb", "fb")],
        require_held_out_model_family=True,
    )


def test_activation_feature_validation_paths() -> None:
    with pytest.raises(ValidationError, match="sample and activation"):
        activation_summary_features([1.0, 2.0])
    summary = activation_summary_features(np.arange(24.0).reshape(2, 3, 4))
    assert summary.shape == (2, 7)
    with pytest.raises(ValidationError, match="two-dimensional"):
        aligned_activation_features(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    with pytest.raises(ValidationError, match="dimensions must match"):
        aligned_activation_features(np.zeros((2, 3)), np.zeros((2, 4)))
    train, test = aligned_activation_features(np.zeros((2, 3)), np.ones((1, 3)))
    assert train.shape == (2, 3)
    assert test.shape == (1, 3)
