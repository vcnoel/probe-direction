from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from eval_awareness_spectral.representations import (
    GraphOperator,
    SpectralDecomposition,
    build_laplacian,
    compare_graphs,
    compare_joint,
    compare_signals_on_reference,
    decompose_graph,
    graph_fourier_metrics,
)
from eval_awareness_spectral.schemas import SpectralComparisonMode, ValidationError


@pytest.fixture
def irregular_graph() -> np.ndarray:
    return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])


def test_random_walk_legacy_is_nonsymmetric_and_basis_is_always_rejected(
    irregular_graph: np.ndarray,
) -> None:
    laplacian = build_laplacian(irregular_graph, GraphOperator.RANDOM_WALK_LEGACY)
    assert not np.allclose(laplacian, laplacian.T)
    with pytest.raises(ValidationError, match="categorically invalid"):
        decompose_graph(irregular_graph, GraphOperator.RANDOM_WALK_LEGACY)
    regular = np.ones((3, 3)) - np.eye(3)
    with pytest.raises(ValidationError, match="categorically invalid"):
        decompose_graph(regular, GraphOperator.RANDOM_WALK_LEGACY)


def test_symmetric_normalized_basis_has_orthogonality_and_small_residual(
    irregular_graph: np.ndarray,
) -> None:
    basis = decompose_graph(irregular_graph)
    assert np.all(np.diff(basis.eigenvalues) >= -1e-12)
    np.testing.assert_allclose(basis.eigenvectors.T @ basis.eigenvectors, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        basis.laplacian @ basis.eigenvectors,
        basis.eigenvectors * basis.eigenvalues,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        (basis.eigenvectors * basis.eigenvalues) @ basis.eigenvectors.T,
        basis.laplacian,
        atol=1e-12,
    )
    with pytest.raises(ValueError):
        basis.eigenvalues[0] = 4.0
    with pytest.raises(ValueError):
        basis.eigenvalues.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        basis.operator = GraphOperator.COMBINATORIAL


def test_metrics_reject_an_unvalidated_basis(irregular_graph: np.ndarray) -> None:
    laplacian = build_laplacian(irregular_graph)
    values, vectors = np.linalg.eigh(laplacian)
    unvalidated = SpectralDecomposition(
        GraphOperator.SYMMETRIC_NORMALIZED, laplacian, values, vectors
    )
    with pytest.raises(ValidationError, match="validated basis"):
        graph_fourier_metrics(np.ones(3), unvalidated)


def test_comparison_modes_are_distinct(irregular_graph: np.ndarray) -> None:
    reference = decompose_graph(irregular_graph)
    target = decompose_graph(np.ones((3, 3)) - np.eye(3))
    graph = compare_graphs(reference, target)
    signal = compare_signals_on_reference(np.arange(3.0), np.arange(3.0) + 1.0, reference)
    joint = compare_joint(np.arange(3.0), np.arange(3.0) + 1.0, reference, target)
    assert graph.mode is SpectralComparisonMode.GRAPH_ONLY
    assert signal.mode is SpectralComparisonMode.FIXED_REFERENCE_SIGNAL
    assert joint.mode is SpectralComparisonMode.JOINT_GRAPH_AND_SIGNAL
    assert len({graph.mode, signal.mode, joint.mode}) == 3


def test_nonfinite_and_asymmetric_graphs_fail() -> None:
    with pytest.raises(ValidationError, match="finite"):
        decompose_graph([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValidationError, match="symmetric"):
        decompose_graph([[0.0, 1.0], [0.0, 0.0]])
