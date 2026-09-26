import numpy as np
import pytest

from eval_awareness_spectral.evaluation.scaling import (
    EvidenceGate,
    enforce_evidence_language,
    estimate_hierarchical_effect,
)
from eval_awareness_spectral.evaluation.transfer import (
    TransferIdentity,
    activation_summary_features,
    aligned_activation_features,
    validate_transfer_identities,
)
from eval_awareness_spectral.schemas import ValidationError


def test_shared_negative_and_source_task_identities_are_rejected() -> None:
    training = [
        TransferIdentity(
            "row-a", "task-a", "negative-a", "family-a", "template-a"
        )
    ]
    with pytest.raises(ValidationError, match="negative-example"):
        validate_transfer_identities(
            training,
            [
                TransferIdentity(
                    "row-b", "task-b", "negative-a", "family-b", "template-b"
                )
            ],
        )
    with pytest.raises(ValidationError, match="source task"):
        validate_transfer_identities(
            training,
            [
                TransferIdentity(
                    "row-c", "task-a", "negative-c", "family-b", "template-c"
                )
            ],
        )

    with pytest.raises(ValidationError, match="template family"):
        validate_transfer_identities(
            training,
            [
                TransferIdentity(
                    "row-d", "task-d", "negative-d", "family-b", "template-a"
                )
            ],
        )


def test_width_independent_summaries_and_aligned_transfer() -> None:
    narrow = activation_summary_features(np.arange(15, dtype=float).reshape(3, 5))
    wide = activation_summary_features(np.arange(33, dtype=float).reshape(3, 11))
    assert narrow.shape == wide.shape == (3, 7)

    train = np.ones((4, 9))
    test = np.zeros((2, 9))
    aligned_train, aligned_test = aligned_activation_features(train, test)
    assert aligned_train.shape == (4, 9)
    assert aligned_test.shape == (2, 9)


def test_insufficient_family_evidence_refuses_law_wording() -> None:
    with pytest.raises(ValidationError, match="scaling-law"):
        enforce_evidence_language(
            "a universal scaling law",
            ["task-a", "task-b"],
            ["model-a"],
            gate=EvidenceGate(minimum_task_families=3, minimum_model_families=2),
        )


def test_adequate_synthetic_hierarchy_returns_uncertainty() -> None:
    task_families = np.repeat(["reasoning", "coding", "knowledge"], 8)
    model_families = np.tile(np.repeat(["a", "b"], 4), 3)
    task_ids = np.array([f"{family}-{index // 2}" for family in ["r", "c", "k"] for index in range(8)])
    task_effect = np.repeat([-1.0, 0.0, 1.0], 8)
    model_effect = np.tile(np.repeat([-0.5, 0.5], 4), 3)
    values = 2.0 + task_effect + model_effect + np.linspace(-0.1, 0.1, 24)
    result = estimate_hierarchical_effect(
        values,
        task_ids,
        task_families,
        model_families,
        n_bootstrap=200,
        seed=5,
    )
    assert result.low < result.estimate < result.high
    assert result.standard_error > 0.0
    assert set(result.task_family_effects) == {"reasoning", "coding", "knowledge"}
    enforce_evidence_language(
        "scaling law",
        task_families,
        model_families,
        gate=EvidenceGate(minimum_task_families=3, minimum_model_families=2),
    )
