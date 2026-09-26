import numpy as np
import pytest

from eval_awareness_spectral.interventions import (
    BidirectionalMediationResult,
    ControlKind,
    ControlResult,
    DirectionalMediationResult,
    GateStatus,
    GoldScorer,
    KeywordScorer,
    MediationAssessment,
    MediationStatus,
    SteeringDirection,
    assess_mediation,
    behavioral_sign_of_life_gate,
)

SUFFICIENCY = SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY
NECESSITY = SteeringDirection.EVALUATION_TO_DEPLOYMENT_NECESSITY


def test_behavioral_gate_marks_weak_instrument_untestable() -> None:
    gate = behavioral_sign_of_life_gate(
        [0.5, 0.51, 0.5, 0.52], [0, 0, 1, 1], minimum_variation=0.05
    )
    assert gate.status is GateStatus.INSTRUMENT_NOT_TESTABLE


def test_keyword_debug_scorer_is_rejected_for_causal_claim() -> None:
    gate = behavioral_sign_of_life_gate([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], minimum_variation=0.5)
    with pytest.raises(ValueError, match="debug-only"):
        assess_mediation(
            baseline_internal_scores=[0.0],
            intervened_internal_scores=[1.0],
            baseline_observations=["ordinary"],
            intervened_observations=["evaluation"],
            endpoint=KeywordScorer(("evaluation",)),
            gate=gate,
            direction=SUFFICIENCY,
            minimum_internal_movement=0.2,
            minimum_behavioral_movement=0.2,
        )


def test_projection_only_movement_is_not_called_mediation() -> None:
    gate = behavioral_sign_of_life_gate([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], minimum_variation=0.5)
    result = assess_mediation(
        baseline_internal_scores=[0.2, 0.2],
        intervened_internal_scores=[0.2, 0.2],
        baseline_observations=[0.0, 0.0],
        intervened_observations=[1.0, 1.0],
        endpoint=GoldScorer(float),
        gate=gate,
        direction=SUFFICIENCY,
        minimum_internal_movement=0.1,
        minimum_behavioral_movement=0.1,
        projection_movement=100.0,
    )
    assert result.status is MediationStatus.NO_INTERNAL_MOVEMENT
    assert result.behavioral_movement is None


def test_internal_movement_without_behavior_is_not_mediation() -> None:
    gate = behavioral_sign_of_life_gate([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], minimum_variation=0.5)
    result = assess_mediation(
        baseline_internal_scores=[0.0],
        intervened_internal_scores=[1.0],
        baseline_observations=[0.0],
        intervened_observations=[0.0],
        endpoint=GoldScorer(float),
        gate=gate,
        direction=SUFFICIENCY,
        minimum_internal_movement=0.5,
        minimum_behavioral_movement=0.5,
    )
    assert result.status is MediationStatus.NO_BEHAVIORAL_MEDIATION


def _direction_result(direction: SteeringDirection) -> DirectionalMediationResult:
    assessment = MediationAssessment(MediationStatus.MEDIATION, 1.0, 1.0, direction)
    return DirectionalMediationResult(direction, assessment)


def _all_controls() -> tuple[ControlResult, ...]:
    return tuple(
        ControlResult(direction, kind, True)
        for direction in SteeringDirection
        for kind in ControlKind
    )


def test_bidirectional_result_requires_both_directions_and_all_controls() -> None:
    complete = BidirectionalMediationResult(
        _direction_result(SUFFICIENCY), _direction_result(NECESSITY), _all_controls()
    )
    assert complete.mediated_bidirectionally

    with pytest.raises(ValueError, match="missing controls"):
        BidirectionalMediationResult(
            _direction_result(SUFFICIENCY),
            _direction_result(NECESSITY),
            _all_controls()[:-1],
        )
    with pytest.raises(ValueError, match="wrong direction"):
        BidirectionalMediationResult(
            _direction_result(NECESSITY), _direction_result(SUFFICIENCY), _all_controls()
        )


def test_necessity_mediation_uses_reverse_sign() -> None:
    gate = behavioral_sign_of_life_gate([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], minimum_variation=0.5)
    result = assess_mediation(
        baseline_internal_scores=[1.0],
        intervened_internal_scores=[0.0],
        baseline_observations=[1.0],
        intervened_observations=[0.0],
        endpoint=GoldScorer(float),
        gate=gate,
        direction=NECESSITY,
        minimum_internal_movement=0.5,
        minimum_behavioral_movement=0.5,
    )
    assert result.status is MediationStatus.MEDIATION
    assert np.isclose(result.internal_movement, 1.0)
