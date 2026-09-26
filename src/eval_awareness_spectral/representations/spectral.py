"""Validated Euclidean graph-Fourier bases and basis-dependent comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..schemas import SpectralComparisonMode, ValidationError
from .attention_graph import GraphOperator, build_laplacian

FloatArray = npt.NDArray[np.float64]


def _immutable_array(value: npt.ArrayLike) -> FloatArray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


@dataclass(frozen=True)
class SpectralDecomposition:
    """Immutable, validated eigendecomposition of a symmetric graph operator."""

    operator: GraphOperator
    laplacian: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "laplacian", _immutable_array(self.laplacian))
        object.__setattr__(self, "eigenvalues", _immutable_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _immutable_array(self.eigenvectors))


@dataclass(frozen=True)
class GraphFourierMetrics:
    spectral_entropy: float
    hfer: float
    spectral_mass_gini: float
    fiedler_value: float


@dataclass(frozen=True)
class GraphOnlyChange:
    mode: SpectralComparisonMode
    eigenvalue_l2: float
    fiedler_delta: float


@dataclass(frozen=True)
class FixedReferenceSignalChange:
    mode: SpectralComparisonMode
    before: GraphFourierMetrics
    after: GraphFourierMetrics


@dataclass(frozen=True)
class JointGraphAndSignalChange:
    mode: SpectralComparisonMode
    reference: GraphFourierMetrics
    target: GraphFourierMetrics
    graph_change: GraphOnlyChange


def _validate_basis(
    laplacian: npt.ArrayLike,
    eigenvalues: npt.ArrayLike,
    eigenvectors: npt.ArrayLike,
    operator: GraphOperator,
) -> SpectralDecomposition:
    if operator is GraphOperator.RANDOM_WALK_LEGACY:
        raise ValidationError(
            "random-walk-legacy is categorically invalid for basis-dependent analysis"
        )
    matrix = np.asarray(laplacian, dtype=np.float64)
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Laplacian must be square")
    size = matrix.shape[0]
    if values.shape != (size,) or vectors.shape != (size, size):
        raise ValidationError("eigendecomposition shapes do not match the Laplacian")
    if not all(np.all(np.isfinite(item)) for item in (matrix, values, vectors)):
        raise ValidationError("eigendecomposition must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValidationError("a Euclidean graph-Fourier basis requires a symmetric operator")
    tolerance = 1e-9 * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if np.any(np.diff(values) < -tolerance):
        raise ValidationError("eigenvalues must be sorted in ascending order")
    identity = np.eye(size)
    if not np.allclose(vectors.T @ vectors, identity, rtol=1e-8, atol=1e-9):
        raise ValidationError("eigenvectors are not Euclidean-orthonormal")
    reconstruction = (vectors * values[None, :]) @ vectors.T
    if not np.allclose(reconstruction, matrix, rtol=1e-8, atol=tolerance):
        raise ValidationError("eigendecomposition does not reconstruct the Laplacian")
    residual = matrix @ vectors - vectors * values[None, :]
    if float(np.linalg.norm(residual, ord="fro")) > tolerance * max(1.0, np.sqrt(size)):
        raise ValidationError("eigenpair residual exceeds tolerance")
    matrix = matrix.copy()
    values = values.copy()
    vectors = vectors.copy()
    return SpectralDecomposition(operator, matrix, values, vectors, _validated=True)


def decompose_graph(
    adjacency: npt.ArrayLike,
    operator: GraphOperator = GraphOperator.SYMMETRIC_NORMALIZED,
) -> SpectralDecomposition:
    """Construct and validate an orthonormal basis for an undirected graph."""

    try:
        selected = GraphOperator(operator)
    except ValueError as error:
        raise ValidationError(f"unsupported graph operator: {operator}") from error
    if selected is GraphOperator.RANDOM_WALK_LEGACY:
        raise ValidationError(
            "random-walk-legacy is categorically invalid for basis-dependent analysis"
        )
    laplacian = build_laplacian(adjacency, selected)
    if not np.allclose(laplacian, laplacian.T, rtol=1e-10, atol=1e-12):
        raise ValidationError("basis operator must be symmetric")
    # eigh is deliberately reached only after graph and operator validation.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return _validate_basis(laplacian, eigenvalues, eigenvectors, selected)


def _require_validated(basis: SpectralDecomposition) -> None:
    if not isinstance(basis, SpectralDecomposition) or not basis._validated:
        raise ValidationError("graph-Fourier metrics require a validated basis")
    _validate_basis(basis.laplacian, basis.eigenvalues, basis.eigenvectors, basis.operator)


def _signal_matrix(signal: npt.ArrayLike, node_count: int) -> FloatArray:
    value = np.asarray(signal, dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != node_count:
        raise ValidationError("signal must have shape (nodes,) or (nodes, features)")
    if not np.all(np.isfinite(value)):
        raise ValidationError("signal must contain only finite values")
    return value


def graph_fourier_metrics(
    signal: npt.ArrayLike,
    basis: SpectralDecomposition,
    *,
    high_frequency_start: int | None = None,
) -> GraphFourierMetrics:
    """Compute entropy, upper-band energy ratio, mass Gini, and Fiedler value."""

    _require_validated(basis)
    value = _signal_matrix(signal, basis.eigenvectors.shape[0])
    coefficients = basis.eigenvectors.T @ value
    mass = np.sum(coefficients**2, axis=1)
    total = float(mass.sum())
    if total <= 0.0:
        probabilities = np.zeros_like(mass)
        entropy = 0.0
        gini = 0.0
        hfer = 0.0
    else:
        probabilities = mass / total
        nonzero = probabilities > 0.0
        normalizer = np.log(len(probabilities)) if len(probabilities) > 1 else 1.0
        entropy = float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / normalizer)
        sorted_mass = np.sort(probabilities)
        count = len(sorted_mass)
        gini = float(
            np.sum((2 * np.arange(1, count + 1) - count - 1) * sorted_mass) / count
        )
        start = count // 2 if high_frequency_start is None else high_frequency_start
        if not 0 <= start <= count:
            raise ValidationError("high_frequency_start must index the spectral basis")
        hfer = float(probabilities[start:].sum())
    fiedler = float(basis.eigenvalues[1]) if len(basis.eigenvalues) > 1 else 0.0
    return GraphFourierMetrics(entropy, hfer, gini, fiedler)


def compare_graphs(
    reference: SpectralDecomposition, target: SpectralDecomposition
) -> GraphOnlyChange:
    """Compare graph spectra without introducing a node signal."""

    _require_validated(reference)
    _require_validated(target)
    if reference.eigenvalues.shape != target.eigenvalues.shape:
        raise ValidationError("graph comparison requires equal node counts")
    delta = target.eigenvalues - reference.eigenvalues
    return GraphOnlyChange(
        SpectralComparisonMode.GRAPH_ONLY,
        float(np.linalg.norm(delta)),
        float((target.eigenvalues[1] - reference.eigenvalues[1]) if len(delta) > 1 else 0.0),
    )


def compare_signals_on_reference(
    before: npt.ArrayLike,
    after: npt.ArrayLike,
    reference: SpectralDecomposition,
) -> FixedReferenceSignalChange:
    """Compare two signals in one fixed reference basis."""

    return FixedReferenceSignalChange(
        SpectralComparisonMode.FIXED_REFERENCE_SIGNAL,
        graph_fourier_metrics(before, reference),
        graph_fourier_metrics(after, reference),
    )


def compare_joint(
    reference_signal: npt.ArrayLike,
    target_signal: npt.ArrayLike,
    reference: SpectralDecomposition,
    target: SpectralDecomposition,
) -> JointGraphAndSignalChange:
    """Compare graph-plus-signal states, keeping basis changes explicit."""

    return JointGraphAndSignalChange(
        SpectralComparisonMode.JOINT_GRAPH_AND_SIGNAL,
        graph_fourier_metrics(reference_signal, reference),
        graph_fourier_metrics(target_signal, target),
        compare_graphs(reference, target),
    )
