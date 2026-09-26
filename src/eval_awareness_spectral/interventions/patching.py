"""Precisely targeted residual-stream patching and control construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
RoleIndices: TypeAlias = Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class TokenTarget:
    """A token selection expressed as explicit indices, half-open spans, and/or roles."""

    indices: tuple[int, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.indices or self.spans or self.roles):
            raise ValueError("a token target must select at least one index, span, or role")
        if any(index < 0 for index in self.indices):
            raise ValueError("token indices must be non-negative")
        if any(start < 0 or stop <= start for start, stop in self.spans):
            raise ValueError("token spans must be non-empty, non-negative half-open intervals")
        if any(not role for role in self.roles):
            raise ValueError("token roles must be non-empty")

    def resolve(self, token_count: int, role_indices: RoleIndices | None = None) -> tuple[int, ...]:
        """Resolve the selection, rejecting missing roles and out-of-range positions."""
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        selected = set(self.indices)
        for start, stop in self.spans:
            if stop > token_count:
                raise ValueError(f"token span ({start}, {stop}) exceeds token count {token_count}")
            selected.update(range(start, stop))
        role_map = role_indices or {}
        for role in self.roles:
            if role not in role_map:
                raise ValueError(f"no token indices supplied for role {role!r}")
            selected.update(role_map[role])
        if any(index < 0 or index >= token_count for index in selected):
            raise ValueError(f"token selection contains an index outside [0, {token_count})")
        if not selected:
            raise ValueError("resolved token selection is empty")
        return tuple(sorted(selected))


@dataclass(frozen=True)
class SubspaceBasis:
    """A finite, full-column-rank hidden-space basis."""

    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("subspace basis must have shape (hidden_size, rank) with nonzero axes")
        if not np.isfinite(values).all():
            raise ValueError("subspace basis must contain only finite values")
        if values.shape[1] > values.shape[0]:
            raise ValueError("subspace rank cannot exceed hidden size")
        if np.linalg.matrix_rank(values) != values.shape[1]:
            raise ValueError("subspace basis columns must be linearly independent")
        immutable = values.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "values", immutable)

    @property
    def hidden_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def rank(self) -> int:
        return int(self.values.shape[1])

    def orthonormal(self) -> FloatArray:
        q, _ = np.linalg.qr(self.values, mode="reduced")
        return np.asarray(q, dtype=np.float64)


@dataclass(frozen=True)
class InterventionTarget:
    """Layers and prompt tokens at which an intervention is permitted."""

    layers: tuple[int, ...]
    tokens: TokenTarget
    subspace: SubspaceBasis | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("at least one layer must be targeted")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("target layers must be unique")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("target layers must be non-negative")


class ControlKind(StrEnum):
    EQUAL_RANK_RANDOM = "equal_rank_random"
    SHUFFLED_LABEL = "shuffled_label"
    PLACEBO_CUE = "placebo_cue"
    FULL_RESIDUAL = "full_residual"


@dataclass(frozen=True)
class ControlDescriptor:
    """Auditable description of a causal intervention control."""

    kind: ControlKind
    target: InterventionTarget
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _residual_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (layers, tokens, hidden_size)")
    if 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def _resolved_target(
    residual: FloatArray, target: InterventionTarget, role_indices: RoleIndices | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if any(layer >= residual.shape[0] for layer in target.layers):
        raise ValueError("target layer is outside the residual tensor")
    tokens = target.tokens.resolve(int(residual.shape[1]), role_indices)
    if target.subspace is not None and target.subspace.hidden_size != residual.shape[2]:
        raise ValueError("subspace hidden size does not match the residual tensor")
    return target.layers, tokens


def patch_residual(
    recipient: ArrayLike,
    donor: ArrayLike,
    target: InterventionTarget,
    *,
    role_indices: RoleIndices | None = None,
) -> FloatArray:
    """Token-aligned replacement of selected residual vectors."""
    recipient_array = _residual_array(recipient, "recipient")
    donor_array = _residual_array(donor, "donor")
    if recipient_array.shape != donor_array.shape:
        raise ValueError("recipient and donor residual tensors must have identical shapes")
    layers, tokens = _resolved_target(recipient_array, target, role_indices)
    result = recipient_array.copy()
    for layer in layers:
        result[layer, list(tokens), :] = donor_array[layer, list(tokens), :]
    return result


def interchange_subspace(
    recipient: ArrayLike,
    donor: ArrayLike,
    target: InterventionTarget,
    *,
    role_indices: RoleIndices | None = None,
) -> FloatArray:
    """Replace only the selected subspace coordinates at aligned tokens."""
    recipient_array = _residual_array(recipient, "recipient")
    donor_array = _residual_array(donor, "donor")
    if recipient_array.shape != donor_array.shape:
        raise ValueError("recipient and donor residual tensors must have identical shapes")
    if target.subspace is None:
        raise ValueError("subspace interchange requires target.subspace")
    layers, tokens = _resolved_target(recipient_array, target, role_indices)
    q = target.subspace.orthonormal()
    result = recipient_array.copy()
    for layer in layers:
        for token in tokens:
            difference = donor_array[layer, token] - recipient_array[layer, token]
            result[layer, token] += (difference @ q) @ q.T
    return result


def equal_rank_random_control(
    target: InterventionTarget, *, rng: np.random.Generator
) -> ControlDescriptor:
    """Build an equal-rank random-subspace control for a subspace target."""
    if target.subspace is None:
        raise ValueError("equal-rank random control requires a subspace target")
    random_values = rng.normal(size=(target.subspace.hidden_size, target.subspace.rank))
    random_basis = SubspaceBasis(random_values)
    control_target = InterventionTarget(target.layers, target.tokens, random_basis)
    return ControlDescriptor(
        ControlKind.EQUAL_RANK_RANDOM,
        control_target,
        {"rank": str(random_basis.rank)},
    )


def shuffled_label_control(target: InterventionTarget) -> ControlDescriptor:
    return ControlDescriptor(ControlKind.SHUFFLED_LABEL, target, {"assignment": "shuffled"})


def placebo_cue_control(target: InterventionTarget) -> ControlDescriptor:
    return ControlDescriptor(ControlKind.PLACEBO_CUE, target, {"cue": "placebo"})


def full_residual_control(target: InterventionTarget) -> ControlDescriptor:
    full_target = InterventionTarget(target.layers, target.tokens, None)
    return ControlDescriptor(ControlKind.FULL_RESIDUAL, full_target, {"scope": "full_hidden"})


# Descriptive aliases used in experiment specifications.
token_aligned_residual_patching = patch_residual
subspace_interchange = interchange_subspace
