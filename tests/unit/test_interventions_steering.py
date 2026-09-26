import numpy as np

from eval_awareness_spectral.interventions import (
    InterventionTarget,
    SteeringDirection,
    SteeringSpec,
    TokenTarget,
    apply_residual_steering,
)


def test_steering_targets_only_selected_layers_and_prompt_tokens() -> None:
    residual = np.zeros((3, 6, 2))
    target = InterventionTarget((0, 2), TokenTarget(roles=("cue",)))
    spec = SteeringSpec(
        target,
        np.array([2.0, -1.0]),
        0.5,
        SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY,
    )

    result = apply_residual_steering(residual, spec, role_indices={"cue": (1, 2)})

    expected = np.zeros_like(residual)
    expected[[[0], [2]], [1, 2], :] = [1.0, -0.5]
    np.testing.assert_array_equal(result, expected)
    # Positions 3--5 model generated/unselected tokens and remain untouched.
    np.testing.assert_array_equal(result[:, 3:], 0.0)


def test_necessity_direction_reverses_steering_sign() -> None:
    residual = np.zeros((1, 2, 2))
    target = InterventionTarget((0,), TokenTarget(indices=(1,)))
    forward = SteeringSpec(
        target,
        np.array([1.0, 2.0]),
        3.0,
        SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY,
    )
    reverse = SteeringSpec(
        target,
        np.array([1.0, 2.0]),
        3.0,
        SteeringDirection.EVALUATION_TO_DEPLOYMENT_NECESSITY,
    )

    np.testing.assert_array_equal(
        apply_residual_steering(residual, reverse),
        -apply_residual_steering(residual, forward),
    )
