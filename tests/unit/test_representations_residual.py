from __future__ import annotations

import numpy as np
import pytest

from eval_awareness_spectral.representations import (
    dirichlet_energy,
    ordinary_dc_ac,
    symmetric_normalized_dc_ac,
    symmetric_normalized_dirichlet_energy,
    symmetrize_adjacency,
    validate_adjacency,
)
from eval_awareness_spectral.schemas import ValidationError


def test_combinatorial_energy_is_invariant_to_uniform_token_shift() -> None:
    graph = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    signal = np.array([[1.0, 3.0], [2.0, -1.0], [4.0, 2.0]])
    shift = np.array([17.0, -4.0])
    assert dirichlet_energy(signal, graph) == pytest.approx(
        dirichlet_energy(signal + shift, graph)
    )


def test_normalized_energy_changes_under_uniform_shift_on_irregular_graph() -> None:
    graph = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    signal = np.array([1.0, 2.0, 4.0])
    before = symmetric_normalized_dirichlet_energy(signal, graph)
    after = symmetric_normalized_dirichlet_energy(signal + 5.0, graph)
    assert after != pytest.approx(before)


def test_lifted_dc_and_ac_norms_sum_and_include_token_factor() -> None:
    signal = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = ordinary_dc_ac(signal)
    mean_norm = np.linalg.norm(signal.mean(axis=0))
    assert np.sqrt(result.dc_squared_norm) == pytest.approx(np.sqrt(3.0) * mean_norm)
    assert result.dc_squared_norm + result.ac_squared_norm == pytest.approx(
        result.total_squared_norm
    )
    np.testing.assert_allclose(result.dc + result.ac, signal)


def test_degree_weighted_dc_ac_is_orthogonal() -> None:
    graph = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    result = symmetric_normalized_dc_ac(np.array([1.0, 2.0, 4.0]), graph)
    assert float(result.dc @ result.ac) == pytest.approx(0.0, abs=1e-12)
    assert result.dc_squared_norm + result.ac_squared_norm == pytest.approx(
        result.total_squared_norm
    )


def test_symmetrization_is_explicit_and_validation_never_changes_input() -> None:
    directed = np.array([[0.0, 2.0], [0.0, 0.0]])
    checked = validate_adjacency(directed)
    np.testing.assert_array_equal(checked, directed)
    with pytest.raises(ValidationError, match="symmetric"):
        validate_adjacency(directed, symmetric=True)
    np.testing.assert_allclose(symmetrize_adjacency(directed), [[0.0, 1.0], [1.0, 0.0]])
