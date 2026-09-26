import numpy as np
import pytest

from eval_awareness_spectral.interventions import (
    ControlKind,
    InterventionTarget,
    SubspaceBasis,
    TokenTarget,
    equal_rank_random_control,
    full_residual_control,
    interchange_subspace,
    patch_residual,
    placebo_cue_control,
    shuffled_label_control,
)


def test_residual_patch_targets_exact_layer_and_role_tokens() -> None:
    recipient = np.zeros((3, 5, 4))
    donor = np.arange(60, dtype=float).reshape(3, 5, 4)
    target = InterventionTarget((1,), TokenTarget(indices=(0,), roles=("cue",)))
    result = patch_residual(recipient, donor, target, role_indices={"cue": (3,)})

    expected = np.zeros_like(recipient)
    expected[1, [0, 3]] = donor[1, [0, 3]]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(recipient, np.zeros_like(recipient))


def test_subspace_interchange_is_rank_limited() -> None:
    recipient = np.zeros((2, 3, 3))
    donor = np.ones_like(recipient)
    basis = SubspaceBasis(np.array([[1.0], [0.0], [0.0]]))
    target = InterventionTarget((0,), TokenTarget(spans=((1, 2),)), basis)

    result = interchange_subspace(recipient, donor, target)

    np.testing.assert_array_equal(result[0, 1], [1.0, 0.0, 0.0])
    assert np.count_nonzero(result) == 1


def test_basis_rejects_bad_shape_rank_and_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        SubspaceBasis(np.ones(3))
    with pytest.raises(ValueError, match="independent"):
        SubspaceBasis(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(ValueError, match="finite"):
        SubspaceBasis(np.array([[np.inf], [0.0]]))


def test_controls_are_identified_and_random_control_has_equal_rank() -> None:
    target = InterventionTarget(
        (0,), TokenTarget(indices=(1,)), SubspaceBasis(np.eye(4)[:, :2])
    )
    controls = (
        equal_rank_random_control(target, rng=np.random.default_rng(4)),
        shuffled_label_control(target),
        placebo_cue_control(target),
        full_residual_control(target),
    )

    assert {control.kind for control in controls} == set(ControlKind)
    assert controls[0].target.subspace is not None
    assert target.subspace is not None
    assert controls[0].target.subspace.rank == target.subspace.rank
    assert controls[-1].target.subspace is None
