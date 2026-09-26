"""Behavioral endpoints, sign-of-life gates, and mediation result validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .patching import ControlKind
from .steering import SteeringDirection

FloatArray: TypeAlias = NDArray[np.float64]
Observation: TypeAlias = object


@runtime_checkable
class BehaviorEndpoint(Protocol):
    """Interface for a scored behavioral outcome."""

    @property
    def scientific(self) -> bool: ...

    @property
    def name(self) -> str: ...

    def score(self, observations: Sequence[Observation]) -> FloatArray: ...


@dataclass(frozen=True)
class GoldScorer:
    """A scientific endpoint backed by task gold labels or an exact rubric."""

    scoring_function: Callable[[Observation], float]
    name: str = "gold"

    @property
    def scientific(self) -> bool:
        return True

    def score(self, observations: Sequence[Observation]) -> FloatArray:
        return _checked_scores([self.scoring_function(item) for item in observations], self.name)


@dataclass(frozen=True)
class ValidatedJudgeScorer:
    """A judge endpoint whose independent validation must already have passed."""

    scoring_function: Callable[[Observation], float]
    validation_passed: bool
    name: str = "validated_judge"

    @property
    def scientific(self) -> bool:
        return self.validation_passed

    def score(self, observations: Sequence[Observation]) -> FloatArray:
        return _checked_scores([self.scoring_function(item) for item in observations], self.name)


@dataclass(frozen=True)
class KeywordScorer:
    """Keyword matching is useful for debugging, never for a causal claim."""

    keywords: tuple[str, ...]
    name: str = "keyword_debug_only"

    def __post_init__(self) -> None:
        if not self.keywords or any(not keyword for keyword in self.keywords):
            raise ValueError("at least one non-empty keyword is required")

    @property
    def scientific(self) -> bool:
        return False

    def score(self, observations: Sequence[Observation]) -> FloatArray:
        lowered = tuple(keyword.casefold() for keyword in self.keywords)
        values = [float(any(keyword in str(item).casefold() for keyword in lowered)) for item in observations]
        return np.asarray(values, dtype=np.float64)


def _checked_scores(values: ArrayLike, name: str) -> FloatArray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(f"{name} must produce a non-empty one-dimensional score array")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} produced non-finite scores")
    return scores


class GateStatus(StrEnum):
    TESTABLE = "testable"
    INSTRUMENT_NOT_TESTABLE = "instrument_not_testable"


@dataclass(frozen=True)
class BehavioralGateResult:
    status: GateStatus
    condition_linked_variation: float
    threshold: float


def behavioral_sign_of_life_gate(
    outcomes: ArrayLike,
    conditions: ArrayLike,
    *,
    minimum_variation: float,
    endpoint_validated: bool = True,
) -> BehavioralGateResult:
    """Check that a validated endpoint varies with the known behavior condition."""
    scores = _checked_scores(outcomes, "sign-of-life outcomes")
    labels = np.asarray(conditions)
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("conditions must be one-dimensional and align with outcomes")
    if minimum_variation < 0.0 or not np.isfinite(minimum_variation):
        raise ValueError("minimum_variation must be finite and non-negative")
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError("sign-of-life conditions must contain exactly two groups")
    group_a = scores[labels == unique[0]]
    group_b = scores[labels == unique[1]]
    variation = float(abs(np.mean(group_b) - np.mean(group_a)))
    testable = endpoint_validated and variation >= minimum_variation
    return BehavioralGateResult(
        GateStatus.TESTABLE if testable else GateStatus.INSTRUMENT_NOT_TESTABLE,
        variation,
        minimum_variation,
    )


class MediationStatus(StrEnum):
    MEDIATION = "mediation"
    NO_INTERNAL_MOVEMENT = "no_internal_movement"
    NO_BEHAVIORAL_MEDIATION = "no_behavioral_mediation"
    INSTRUMENT_NOT_TESTABLE = "instrument_not_testable"


@dataclass(frozen=True)
class MediationAssessment:
    status: MediationStatus
    internal_movement: float
    behavioral_movement: float | None
    direction: SteeringDirection

    @property
    def mediated(self) -> bool:
        return self.status is MediationStatus.MEDIATION


def assess_mediation(
    *,
    baseline_internal_scores: ArrayLike,
    intervened_internal_scores: ArrayLike,
    baseline_observations: Sequence[Observation],
    intervened_observations: Sequence[Observation],
    endpoint: BehaviorEndpoint,
    gate: BehavioralGateResult,
    direction: SteeringDirection,
    minimum_internal_movement: float,
    minimum_behavioral_movement: float,
    projection_movement: float | None = None,
) -> MediationAssessment:
    """Assess mediation, requiring internal-score movement before behavior.

    ``projection_movement`` is accepted for audit trails but cannot satisfy the
    internal criterion: only the supplied, independently defined internal score
    is used to establish movement.
    """
    del projection_movement
    if not endpoint.scientific:
        raise ValueError("debug-only or unvalidated scorers are rejected for causal claims")
    if minimum_internal_movement < 0.0 or minimum_behavioral_movement < 0.0:
        raise ValueError("movement thresholds must be non-negative")
    if gate.status is GateStatus.INSTRUMENT_NOT_TESTABLE:
        return MediationAssessment(
            MediationStatus.INSTRUMENT_NOT_TESTABLE, 0.0, None, direction
        )
    baseline_internal = _checked_scores(baseline_internal_scores, "baseline internal scores")
    intervened_internal = _checked_scores(
        intervened_internal_scores, "intervened internal scores"
    )
    if baseline_internal.shape != intervened_internal.shape:
        raise ValueError("baseline and intervened internal scores must align")
    signed_internal = direction.sign * float(
        np.mean(intervened_internal) - np.mean(baseline_internal)
    )
    if signed_internal < minimum_internal_movement:
        return MediationAssessment(
            MediationStatus.NO_INTERNAL_MOVEMENT, signed_internal, None, direction
        )
    baseline_behavior = endpoint.score(baseline_observations)
    intervened_behavior = endpoint.score(intervened_observations)
    if baseline_behavior.shape != intervened_behavior.shape:
        raise ValueError("baseline and intervened behavioral scores must align")
    signed_behavior = direction.sign * float(
        np.mean(intervened_behavior) - np.mean(baseline_behavior)
    )
    status = (
        MediationStatus.MEDIATION
        if signed_behavior >= minimum_behavioral_movement
        else MediationStatus.NO_BEHAVIORAL_MEDIATION
    )
    return MediationAssessment(status, signed_internal, signed_behavior, direction)


@dataclass(frozen=True)
class DirectionalMediationResult:
    direction: SteeringDirection
    assessment: MediationAssessment

    def __post_init__(self) -> None:
        if self.assessment.direction is not self.direction:
            raise ValueError("result direction does not match assessment direction")


@dataclass(frozen=True)
class ControlResult:
    direction: SteeringDirection
    kind: ControlKind
    passed: bool


_REQUIRED_CONTROLS = frozenset(ControlKind)


@dataclass(frozen=True)
class BidirectionalMediationResult:
    """A complete claim package: both directions and every control per direction."""

    sufficiency: DirectionalMediationResult
    necessity: DirectionalMediationResult
    controls: tuple[ControlResult, ...]

    def __post_init__(self) -> None:
        if self.sufficiency.direction is not SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY:
            raise ValueError("sufficiency result has the wrong direction")
        if self.necessity.direction is not SteeringDirection.EVALUATION_TO_DEPLOYMENT_NECESSITY:
            raise ValueError("necessity result has the wrong direction")
        seen: dict[SteeringDirection, set[ControlKind]] = {
            SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY: set(),
            SteeringDirection.EVALUATION_TO_DEPLOYMENT_NECESSITY: set(),
        }
        for control in self.controls:
            if control.kind in seen[control.direction]:
                raise ValueError(
                    f"duplicate {control.kind.value} control for {control.direction.value}"
                )
            seen[control.direction].add(control.kind)
        for direction, kinds in seen.items():
            missing = _REQUIRED_CONTROLS - kinds
            if missing:
                names = ", ".join(sorted(kind.value for kind in missing))
                raise ValueError(f"missing controls for {direction.value}: {names}")
        if not all(control.passed for control in self.controls):
            raise ValueError("all control results must pass before reporting a bidirectional claim")

    @property
    def mediated_bidirectionally(self) -> bool:
        return self.sufficiency.assessment.mediated and self.necessity.assessment.mediated


def validate_bidirectional_results(
    results: Mapping[SteeringDirection, DirectionalMediationResult],
    controls: Sequence[ControlResult],
) -> BidirectionalMediationResult:
    """Validate and return a complete bidirectional result package."""
    required = set(SteeringDirection)
    if set(results) != required:
        raise ValueError("exactly one sufficiency and one necessity result are required")
    return BidirectionalMediationResult(
        results[SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY],
        results[SteeringDirection.EVALUATION_TO_DEPLOYMENT_NECESSITY],
        tuple(controls),
    )
