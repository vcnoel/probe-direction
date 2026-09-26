"""Dependency-free adapters for local manifests and NumPy activations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from ..claims import validate_result_manifest
from ..data.manifests import read_manifest
from ..representations import (
    DCACDecomposition,
    GraphFourierMetrics,
    SpectralDecomposition,
    dirichlet_energy,
    graph_fourier_metrics,
    ordinary_dc_ac,
)
from ..schemas import PromptRecord, ValidationError

FloatArray: TypeAlias = npt.NDArray[np.float64]
ActivationLayout: TypeAlias = Literal[
    "tokens-hidden",
    "sequences-tokens-hidden",
    "layers-tokens-hidden",
    "sequences-layers-tokens-hidden",
]


def _immutable_float_array(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValidationError("activations must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ActivationRecord:
    """One activation vector with explicit sequence, layer, and token provenance."""

    sequence_id: str
    layer: int
    token_index: int
    values: FloatArray

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValidationError("activation sequence_id must be non-empty")
        if self.layer < 0 or self.token_index < 0:
            raise ValidationError("activation layer and token_index must be non-negative")
        values = _immutable_float_array(self.values)
        if values.ndim != 1 or values.size == 0:
            raise ValidationError("each activation must be a non-empty vector")
        object.__setattr__(self, "values", values)

    @property
    def activation(self) -> FloatArray:
        """Compatibility alias used by activation-dataset-like containers."""

        return self.values


@dataclass(frozen=True)
class NumpyActivationBatch:
    """Canonical ``(sequence, layer, token, hidden)`` activation tensor."""

    values: FloatArray
    sequence_ids: tuple[str, ...]
    layers: tuple[int, ...]
    token_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        values = _immutable_float_array(self.values)
        if values.ndim != 4 or 0 in values.shape:
            raise ValidationError(
                "activation batch must have shape (sequences, layers, tokens, hidden_size)"
            )
        expected = values.shape[:3]
        actual = (len(self.sequence_ids), len(self.layers), len(self.token_indices))
        if actual != expected:
            raise ValidationError("activation provenance lengths do not match tensor axes")
        if len(set(self.sequence_ids)) != len(self.sequence_ids) or any(
            not item for item in self.sequence_ids
        ):
            raise ValidationError("sequence IDs must be non-empty and unique")
        if len(set(self.layers)) != len(self.layers) or any(item < 0 for item in self.layers):
            raise ValidationError("layers must be unique non-negative integers")
        if len(set(self.token_indices)) != len(self.token_indices) or any(
            item < 0 for item in self.token_indices
        ):
            raise ValidationError("token indices must be unique non-negative integers")
        object.__setattr__(self, "values", values)

    def rows(self) -> Iterator[ActivationRecord]:
        """Yield vector rows without discarding any axis provenance."""

        for sequence_position, sequence_id in enumerate(self.sequence_ids):
            for layer_position, layer in enumerate(self.layers):
                for token_position, token_index in enumerate(self.token_indices):
                    yield ActivationRecord(
                        sequence_id,
                        layer,
                        token_index,
                        self.values[sequence_position, layer_position, token_position],
                    )

    def signal(self, sequence_id: str, layer: int) -> FloatArray:
        """Return the token-by-hidden signal consumed by representation functions."""

        try:
            sequence_position = self.sequence_ids.index(sequence_id)
            layer_position = self.layers.index(layer)
        except ValueError as error:
            raise ValidationError(
                f"unknown activation selection: sequence_id={sequence_id!r}, layer={layer}"
            ) from error
        return np.asarray(self.values[sequence_position, layer_position], dtype=np.float64)

    def ordinary_dc_ac(self, sequence_id: str, layer: int) -> DCACDecomposition:
        return ordinary_dc_ac(self.signal(sequence_id, layer))

    def dirichlet_energy(self, sequence_id: str, layer: int, adjacency: npt.ArrayLike) -> float:
        return dirichlet_energy(self.signal(sequence_id, layer), adjacency)

    def graph_fourier_metrics(
        self,
        sequence_id: str,
        layer: int,
        basis: SpectralDecomposition,
        *,
        high_frequency_start: int | None = None,
    ) -> GraphFourierMetrics:
        return graph_fourier_metrics(
            self.signal(sequence_id, layer),
            basis,
            high_frequency_start=high_frequency_start,
        )


def load_prompt_manifest(path: str | Path) -> list[PromptRecord]:
    """Load and validate a local JSONL prompt manifest."""

    return read_manifest(path)


def load_results_manifest(path: str | Path) -> dict[str, object]:
    """Load a local result manifest and verify every referenced checksum."""

    value = validate_result_manifest(path)
    return dict(value)


def adapt_numpy_activations(
    activations: npt.ArrayLike,
    *,
    layout: ActivationLayout | None = None,
    sequence_ids: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    token_indices: Sequence[int] | None = None,
) -> NumpyActivationBatch:
    """Adapt NumPy-like values to a provenance-bearing representation batch.

    Two-dimensional input means ``(tokens, hidden)``. Three-dimensional input
    defaults to ``(sequences, tokens, hidden)``; pass ``layout="layers-tokens-hidden"``
    for a single sequence with several layers. Four-dimensional input is canonical.
    """

    value = np.asarray(activations, dtype=np.float64)
    selected_layout: ActivationLayout
    if layout is None:
        inferred: dict[int, ActivationLayout] = {
            2: "tokens-hidden",
            3: "sequences-tokens-hidden",
            4: "sequences-layers-tokens-hidden",
        }
        try:
            selected_layout = inferred[value.ndim]
        except KeyError as error:
            raise ValidationError("activations must have two, three, or four dimensions") from error
    else:
        selected_layout = layout

    expected_dimensions = {
        "tokens-hidden": 2,
        "sequences-tokens-hidden": 3,
        "layers-tokens-hidden": 3,
        "sequences-layers-tokens-hidden": 4,
    }
    if value.ndim != expected_dimensions[selected_layout]:
        raise ValidationError(f"layout {selected_layout!r} does not match activation dimensions")
    if selected_layout == "tokens-hidden":
        canonical = value[None, None, :, :]
    elif selected_layout == "sequences-tokens-hidden":
        canonical = value[:, None, :, :]
    elif selected_layout == "layers-tokens-hidden":
        canonical = value[None, :, :, :]
    else:
        canonical = value
    if 0 in canonical.shape:
        raise ValidationError("activation axes must be non-empty")

    sequences = tuple(sequence_ids or (f"sequence-{index}" for index in range(canonical.shape[0])))
    layer_values = tuple(layers or range(canonical.shape[1]))
    tokens = tuple(token_indices or range(canonical.shape[2]))
    return NumpyActivationBatch(canonical, sequences, layer_values, tokens)


# Explicit validation aliases make intent clear at external boundaries.
validate_local_prompt_manifest = load_prompt_manifest
validate_local_results_manifest = load_results_manifest
