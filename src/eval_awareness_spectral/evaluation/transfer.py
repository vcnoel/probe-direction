"""Identity audits and feature construction for representation transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eval_awareness_spectral.evaluation.statistics import finite_array
from eval_awareness_spectral.schemas import ValidationError


@dataclass(frozen=True, slots=True)
class TransferIdentity:
    """Provenance identities whose overlap would invalidate transfer evidence."""

    row_id: str
    source_task_id: str
    negative_example_id: str | None = None
    model_family: str | None = None
    template_family: str | None = None


def _non_null(items: list[TransferIdentity], attribute: str) -> set[str]:
    return {
        value
        for item in items
        if (value := getattr(item, attribute)) is not None and isinstance(value, str)
    }


def validate_transfer_identities(
    training: list[TransferIdentity] | tuple[TransferIdentity, ...],
    evaluation: list[TransferIdentity] | tuple[TransferIdentity, ...],
    *,
    require_held_out_model_family: bool = False,
) -> None:
    """Reject overlap in rows, tasks, negatives, and optionally model families."""

    train = list(training)
    test = list(evaluation)
    if not train or not test:
        raise ValidationError("training and evaluation identities must not be empty")
    if any(
        not isinstance(item.template_family, str) or not item.template_family
        for item in [*train, *test]
    ):
        raise ValidationError(
            "template-disjoint transfer requires template_family on every identity"
        )
    checks = (
        ("row_id", "row ID"),
        ("source_task_id", "source task"),
        ("template_family", "template family"),
        ("negative_example_id", "negative-example identity"),
    )
    for attribute, label in checks:
        overlap = _non_null(train, attribute) & _non_null(test, attribute)
        if overlap:
            raise ValidationError(f"transfer shares {label}: {sorted(overlap)!r}")
    if require_held_out_model_family:
        train_families = _non_null(train, "model_family")
        test_families = _non_null(test, "model_family")
        if not train_families or not test_families:
            raise ValidationError("held-out-family transfer requires model_family on every side")
        overlap = train_families & test_families
        if overlap:
            raise ValidationError(f"transfer shares model family: {sorted(overlap)!r}")


def activation_summary_features(activations: ArrayLike) -> NDArray[np.float64]:
    """Map arbitrary-width activation rows to seven width-independent summaries.

    For ``(samples, ..., width)`` input, every axis after the sample axis is
    summarized. This is appropriate for unaligned architectures; it does not claim
    neuron-level correspondence.
    """

    array = finite_array(activations, name="activations")
    if array.ndim < 2:
        raise ValidationError("activations must have a sample and activation dimension")
    flattened = array.reshape(array.shape[0], -1)
    rms = np.sqrt(np.mean(np.square(flattened), axis=1))
    return np.column_stack(
        (
            np.mean(flattened, axis=1),
            np.std(flattened, axis=1),
            np.min(flattened, axis=1),
            np.max(flattened, axis=1),
            rms,
            np.quantile(flattened, 0.25, axis=1),
            np.quantile(flattened, 0.75, axis=1),
        )
    )


def aligned_activation_features(
    training_activations: ArrayLike,
    evaluation_activations: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Explicitly preserve coordinates when activations have a declared alignment."""

    training = finite_array(training_activations, name="training_activations")
    evaluation = finite_array(evaluation_activations, name="evaluation_activations")
    if training.ndim != 2 or evaluation.ndim != 2:
        raise ValidationError("aligned activations must be two-dimensional")
    if training.shape[1] != evaluation.shape[1]:
        raise ValidationError("aligned activation dimensions must match")
    return training, evaluation


validate_transfer_split = validate_transfer_identities
