"""Validation and explicit symmetrization for attention-derived graphs."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ..schemas import ValidationError

FloatArray = npt.NDArray[np.float64]


class GraphOperator(StrEnum):
    """Supported graph operators.

    ``RANDOM_WALK_LEGACY`` is retained for scalar legacy calculations, but it does
    not define a Euclidean-orthonormal Fourier basis on an irregular graph.
    """

    SYMMETRIC_NORMALIZED = "symmetric-normalized"
    COMBINATORIAL = "combinatorial"
    RANDOM_WALK_LEGACY = "random-walk-legacy"


def validate_adjacency(adjacency: npt.ArrayLike, *, symmetric: bool = False) -> FloatArray:
    """Return a finite, nonnegative, square adjacency matrix.

    Symmetry is checked only when explicitly requested; this function never alters
    the input graph.
    """

    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("adjacency must be a square rank-2 matrix")
    if matrix.shape[0] == 0:
        raise ValidationError("adjacency must contain at least one node")
    if not np.all(np.isfinite(matrix)):
        raise ValidationError("adjacency must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValidationError("adjacency weights must be nonnegative")
    if symmetric and not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValidationError("adjacency must be symmetric; symmetrize it explicitly")
    return matrix.copy()


def symmetrize_adjacency(adjacency: npt.ArrayLike) -> FloatArray:
    """Explicitly replace a directed graph by its arithmetic-mean symmetrization."""

    matrix = validate_adjacency(adjacency)
    result = 0.5 * (matrix + matrix.T)
    if not np.all(np.isfinite(result)):
        raise ValidationError("symmetrization produced nonfinite values")
    return result


def build_laplacian(
    adjacency: npt.ArrayLike,
    operator: GraphOperator = GraphOperator.SYMMETRIC_NORMALIZED,
) -> FloatArray:
    """Build the selected Laplacian without silently symmetrizing the graph."""

    try:
        selected = GraphOperator(operator)
    except ValueError as error:
        raise ValidationError(f"unsupported graph operator: {operator}") from error
    matrix = validate_adjacency(adjacency, symmetric=True)
    degree = matrix.sum(axis=1)
    if selected is GraphOperator.COMBINATORIAL:
        laplacian = np.diag(degree) - matrix
    elif selected is GraphOperator.SYMMETRIC_NORMALIZED:
        inverse_sqrt = np.zeros_like(degree)
        positive = degree > 0.0
        inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian = np.eye(matrix.shape[0]) - inverse_sqrt[:, None] * matrix * inverse_sqrt[None, :]
        # The conventional normalized Laplacian assigns zero rows/columns to isolates.
        laplacian[~positive, ~positive] = 0.0
    else:
        inverse = np.zeros_like(degree)
        positive = degree > 0.0
        inverse[positive] = 1.0 / degree[positive]
        laplacian = np.eye(matrix.shape[0]) - inverse[:, None] * matrix
        laplacian[~positive, ~positive] = 0.0
    if not np.all(np.isfinite(laplacian)):
        raise ValidationError("Laplacian contains nonfinite values")
    return np.asarray(laplacian, dtype=np.float64)
