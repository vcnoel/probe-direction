"""Leakage-safe probe, transfer, selectivity, and scaling evaluation APIs."""

from eval_awareness_spectral.evaluation.probes import (
    NestedCVResult,
    PermutationResult,
    ProbeConfig,
    ProbeEvaluation,
    build_probe_pipeline,
    fit_development_evaluate_confirmation,
    full_pipeline_permutation_test,
    nested_group_cv,
)
from eval_awareness_spectral.evaluation.scaling import (
    EvidenceGate,
    HierarchicalEffectEstimate,
    enforce_evidence_language,
    estimate_hierarchical_effect,
    hierarchical_effect,
)
from eval_awareness_spectral.evaluation.selectivity import (
    SelectivityEstimate,
    paired_decodability_selectivity,
    paired_selectivity,
    selectivity_interaction,
)
from eval_awareness_spectral.evaluation.statistics import (
    BootstrapInterval,
    above_chance_magnitude,
    finite_array,
    orientation_invariant_decodability,
    task_bootstrap_ci,
)
from eval_awareness_spectral.evaluation.transfer import (
    TransferIdentity,
    activation_summary_features,
    aligned_activation_features,
    validate_transfer_identities,
    validate_transfer_split,
)

__all__ = [
    "BootstrapInterval",
    "EvidenceGate",
    "HierarchicalEffectEstimate",
    "NestedCVResult",
    "PermutationResult",
    "ProbeConfig",
    "ProbeEvaluation",
    "SelectivityEstimate",
    "TransferIdentity",
    "above_chance_magnitude",
    "activation_summary_features",
    "aligned_activation_features",
    "build_probe_pipeline",
    "enforce_evidence_language",
    "estimate_hierarchical_effect",
    "finite_array",
    "fit_development_evaluate_confirmation",
    "full_pipeline_permutation_test",
    "hierarchical_effect",
    "nested_group_cv",
    "orientation_invariant_decodability",
    "paired_decodability_selectivity",
    "paired_selectivity",
    "selectivity_interaction",
    "task_bootstrap_ci",
    "validate_transfer_identities",
    "validate_transfer_split",
]
