"""Target-restricted residual steering for bidirectional causal tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .patching import InterventionTarget, RoleIndices

FloatArray: TypeAlias = NDArray[np.float64]


class SteeringDirection(StrEnum):
    """Scientific direction and interpretation of a steering intervention."""

    DEPLOYMENT_TO_EVALUATION_SUFFICIENCY = "deployment_to_evaluation_sufficiency"
    EVALUATION_TO_DEPLOYMENT_NECESSITY = "evaluation_to_deployment_necessity"

    @property
    def sign(self) -> float:
        if self is SteeringDirection.DEPLOYMENT_TO_EVALUATION_SUFFICIENCY:
            return 1.0
        return -1.0


@dataclass(frozen=True)
class SteeringSpec:
    target: InterventionTarget
    vector: FloatArray
    strength: float
    direction: SteeringDirection

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("steering vector must be a non-empty one-dimensional array")
        if not np.isfinite(vector).all() or not np.isfinite(self.strength):
            raise ValueError("steering vector and strength must be finite")
        if np.linalg.norm(vector) == 0.0:
            raise ValueError("steering vector must be nonzero")
        immutable = vector.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "vector", immutable)


def apply_residual_steering(
    residual: ArrayLike,
    spec: SteeringSpec,
    *,
    role_indices: RoleIndices | None = None,
) -> FloatArray:
    """Apply steering at exactly the selected existing tokens and layers.

    Token roles are resolved against the caller-supplied prompt-token map. No
    generated or otherwise unselected position is inferred or modified.
    """
    array = np.asarray(residual, dtype=np.float64)
    if array.ndim != 3 or 0 in array.shape:
        raise ValueError("residual must have shape (layers, tokens, hidden_size)")
    if not np.isfinite(array).all():
        raise ValueError("residual must contain only finite values")
    if spec.vector.shape[0] != array.shape[2]:
        raise ValueError("steering vector hidden size does not match residual tensor")
    if any(layer >= array.shape[0] for layer in spec.target.layers):
        raise ValueError("target layer is outside the residual tensor")
    if spec.target.subspace is not None:
        if spec.target.subspace.hidden_size != array.shape[2]:
            raise ValueError("target subspace hidden size does not match residual tensor")
        q = spec.target.subspace.orthonormal()
        vector = (spec.vector @ q) @ q.T
    else:
        vector = spec.vector
    tokens = spec.target.tokens.resolve(int(array.shape[1]), role_indices)
    result = array.copy()
    delta = spec.direction.sign * spec.strength * vector
    for layer in spec.target.layers:
        result[layer, list(tokens), :] += delta
    return result


residual_steering = apply_residual_steering
