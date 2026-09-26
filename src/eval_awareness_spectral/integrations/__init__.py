"""Dependency-free local adapters for activations and manifests."""

from .standalone import (
    ActivationLayout,
    ActivationRecord,
    NumpyActivationBatch,
    adapt_numpy_activations,
    load_prompt_manifest,
    load_results_manifest,
    validate_local_prompt_manifest,
    validate_local_results_manifest,
)

__all__ = [
    "ActivationLayout",
    "ActivationRecord",
    "NumpyActivationBatch",
    "adapt_numpy_activations",
    "load_prompt_manifest",
    "load_results_manifest",
    "validate_local_prompt_manifest",
    "validate_local_results_manifest",
]
