"""Residual-stream energies and graph-aware DC/AC decompositions.

Combinatorial energy is invariant to adding an ordinary token-uniform shift. In
contrast, symmetric-normalized metrics use the degree-weighted null mode
``sqrt(degree)`` and are therefore *not* ordinary-DC-invariant on irregular graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..schemas import ValidationError
from .attention_graph import GraphOperator, build_laplacian, validate_adjacency

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DCACDecomposition:
    """A signal split into lifted DC and orthogonal AC arrays of identical shape."""

    dc: FloatArray
    ac: FloatArray
    total_squared_norm: float
    dc_squared_norm: float
    ac_squared_norm: float

    def __post_init__(self) -> None:
        self.dc.setflags(write=False)
        self.ac.setflags(write=False)


def _signal(signal: npt.ArrayLike, node_count: int) -> tuple[FloatArray, bool]:
    value = np.asarray(signal, dtype=np.float64)
    was_vector = value.ndim == 1
    if was_vector:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != node_count:
        raise ValidationError("residual signal must have shape (nodes,) or (nodes, features)")
    if not np.all(np.isfinite(value)):
        raise ValidationError("residual signal must contain only finite values")
    return value, was_vector


def _restore(value: FloatArray, was_vector: bool) -> FloatArray:
    result = value[:, 0] if was_vector else value
    return np.asarray(result.copy(), dtype=np.float64)


def dirichlet_energy(signal: npt.ArrayLike, adjacency: npt.ArrayLike) -> float:
    """Return fixed-combinatorial-graph energy ``tr(X.T (D-A) X)``."""

    matrix = validate_adjacency(adjacency, symmetric=True)
    value, _ = _signal(signal, matrix.shape[0])
    laplacian = build_laplacian(matrix, GraphOperator.COMBINATORIAL)
    energy = float(np.sum(value * (laplacian @ value)))
    if not np.isfinite(energy):
        raise ValidationError("Dirichlet energy is nonfinite")
    tolerance = 1e-10 * max(1.0, float(np.sum(value**2)))
    if energy < -tolerance:
        raise ValidationError("Dirichlet energy is unexpectedly negative")
    return max(0.0, energy)


def symmetric_normalized_dirichlet_energy(
    signal: npt.ArrayLike, adjacency: npt.ArrayLike
) -> float:
    """Return normalized energy; unlike raw energy, this is not ordinary-DC-invariant."""

    matrix = validate_adjacency(adjacency, symmetric=True)
    value, _ = _signal(signal, matrix.shape[0])
    laplacian = build_laplacian(matrix, GraphOperator.SYMMETRIC_NORMALIZED)
    energy = float(np.sum(value * (laplacian @ value)))
    if not np.isfinite(energy):
        raise ValidationError("normalized Dirichlet energy is nonfinite")
    tolerance = 1e-10 * max(1.0, float(np.sum(value**2)))
    if energy < -tolerance:
        raise ValidationError("normalized Dirichlet energy is unexpectedly negative")
    return max(0.0, energy)


def _decompose_on_mode(
    signal: npt.ArrayLike, mode: FloatArray
) -> DCACDecomposition:
    value, was_vector = _signal(signal, len(mode))
    norm = float(np.linalg.norm(mode))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValidationError("DC mode must have positive finite norm")
    unit_mode = mode / norm
    coefficients = unit_mode @ value
    dc_matrix = unit_mode[:, None] * coefficients[None, :]
    ac_matrix = value - dc_matrix
    total = float(np.sum(value**2))
    dc_norm = float(np.sum(dc_matrix**2))
    ac_norm = float(np.sum(ac_matrix**2))
    tolerance = 1e-10 * max(1.0, total)
    if abs(total - dc_norm - ac_norm) > tolerance:
        raise ValidationError("DC and AC squared norms do not reconstruct the total")
    return DCACDecomposition(
        _restore(dc_matrix, was_vector),
        _restore(ac_matrix, was_vector),
        total,
        dc_norm,
        ac_norm,
    )


def ordinary_dc_ac(signal: npt.ArrayLike) -> DCACDecomposition:
    """Split around the ordinary token mean, lifting it to all ``n`` tokens.

    Consequently ``||DC_lifted||_F = sqrt(n) * ||token_mean||_2`` rather than the
    norm of the unlifted feature mean.
    """

    value = np.asarray(signal)
    if value.ndim not in (1, 2) or value.shape[0] == 0:
        raise ValidationError("residual signal must have a nonempty token dimension")
    return _decompose_on_mode(signal, np.ones(value.shape[0], dtype=np.float64))


def symmetric_normalized_dc_ac(
    signal: npt.ArrayLike, adjacency: npt.ArrayLike
) -> DCACDecomposition:
    """Split on the normalized-Laplacian null mode proportional to ``sqrt(degree)``."""

    matrix = validate_adjacency(adjacency, symmetric=True)
    degree = matrix.sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValidationError("degree-weighted DC decomposition requires positive degrees")
    return _decompose_on_mode(signal, np.sqrt(degree))
