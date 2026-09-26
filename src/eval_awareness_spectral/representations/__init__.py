"""Validated graph and residual representations for CPU-only spectral analysis."""

from .attention_graph import (
    GraphOperator,
    build_laplacian,
    symmetrize_adjacency,
    validate_adjacency,
)
from .residual import (
    DCACDecomposition,
    dirichlet_energy,
    ordinary_dc_ac,
    symmetric_normalized_dc_ac,
    symmetric_normalized_dirichlet_energy,
)
from .spectral import (
    FixedReferenceSignalChange,
    GraphFourierMetrics,
    GraphOnlyChange,
    JointGraphAndSignalChange,
    SpectralDecomposition,
    compare_graphs,
    compare_joint,
    compare_signals_on_reference,
    decompose_graph,
    graph_fourier_metrics,
)

__all__ = [
    "DCACDecomposition",
    "FixedReferenceSignalChange",
    "GraphFourierMetrics",
    "GraphOnlyChange",
    "GraphOperator",
    "JointGraphAndSignalChange",
    "SpectralDecomposition",
    "build_laplacian",
    "compare_graphs",
    "compare_joint",
    "compare_signals_on_reference",
    "decompose_graph",
    "dirichlet_energy",
    "graph_fourier_metrics",
    "ordinary_dc_ac",
    "symmetric_normalized_dc_ac",
    "symmetric_normalized_dirichlet_energy",
    "symmetrize_adjacency",
    "validate_adjacency",
]
