from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eval_awareness_spectral.data import (
    assign_grouped_splits,
    assign_splits,
    build_prompt_record,
    nested_group_development_folds,
)
from eval_awareness_spectral.representations import (
    GraphOperator,
    SpectralDecomposition,
    compare_graphs,
    decompose_graph,
    graph_fourier_metrics,
)
from eval_awareness_spectral.representations.spectral import _validate_basis
from eval_awareness_spectral.schemas import Split, ValidationError


def _adjacency() -> np.ndarray:
    return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]])


def test_basis_validation_rejects_shape_order_orthogonality_and_residual() -> None:
    basis = decompose_graph(_adjacency())
    with pytest.raises(ValidationError, match="square"):
        _validate_basis(np.zeros((2, 3)), np.zeros(2), np.eye(2), basis.operator)
    with pytest.raises(ValidationError, match="shapes"):
        _validate_basis(np.eye(2), np.zeros(1), np.eye(2), basis.operator)
    with pytest.raises(ValidationError, match="finite"):
        _validate_basis(np.eye(2), np.array([0.0, np.nan]), np.eye(2), basis.operator)
    with pytest.raises(ValidationError, match="symmetric"):
        _validate_basis(np.array([[1.0, 1.0], [0.0, 1.0]]), np.ones(2), np.eye(2), basis.operator)
    with pytest.raises(ValidationError, match="sorted"):
        _validate_basis(np.diag([2.0, 1.0]), np.array([2.0, 1.0]), np.eye(2), basis.operator)
    with pytest.raises(ValidationError, match="orthonormal"):
        _validate_basis(np.eye(2), np.ones(2), np.ones((2, 2)), basis.operator)
    with pytest.raises(ValidationError, match="reconstruct"):
        _validate_basis(np.eye(2), np.zeros(2), np.eye(2), basis.operator)


def test_spectral_metrics_zero_one_node_cutoff_and_unvalidated_paths() -> None:
    single = decompose_graph(np.zeros((1, 1)), GraphOperator.COMBINATORIAL)
    metrics = graph_fourier_metrics([0.0], single)
    assert metrics.spectral_entropy == 0.0
    assert metrics.fiedler_value == 0.0
    with pytest.raises(ValidationError, match="high_frequency_start"):
        graph_fourier_metrics([1.0], single, high_frequency_start=2)
    with pytest.raises(ValidationError, match="shape"):
        graph_fourier_metrics([1.0, 2.0], single)
    with pytest.raises(ValidationError, match="finite"):
        graph_fourier_metrics([np.nan], single)
    fake = SpectralDecomposition(
        GraphOperator.COMBINATORIAL, np.zeros((1, 1)), np.zeros(1), np.ones((1, 1))
    )
    with pytest.raises(ValidationError, match="validated basis"):
        graph_fourier_metrics([1.0], fake)


def test_graph_decomposition_operator_and_comparison_shape_errors() -> None:
    with pytest.raises(ValidationError, match="unsupported graph operator"):
        decompose_graph(_adjacency(), "unsupported")  # type: ignore[arg-type]
    small = decompose_graph(np.zeros((1, 1)), GraphOperator.COMBINATORIAL)
    large = decompose_graph(_adjacency())
    with pytest.raises(ValidationError, match="equal node counts"):
        compare_graphs(small, large)


def _record(index: int, source: str, family: str):
    return build_prompt_record(
        row_id=f"row-{index}",
        source_task_id=source,
        template_family=family,
        environment_factors={"cue": bool(index % 2)},
        condition_valence="ordinary",
        content=f"content-{source}",
        rendered_text=f"rendered-{index}",
        task_tokens=[index + 1],
        tokenizer_revision="tok",
        task_span=(0, 1),
        split=Split.TRAIN,
    )


def test_grouped_split_validation_error_paths() -> None:
    with pytest.raises(ValidationError, match="empty"):
        assign_grouped_splits([])
    row = _record(0, "s0", "f0")
    with pytest.raises(ValidationError, match="repeated"):
        assign_grouped_splits([row, replace(row)])
    with pytest.raises(ValidationError, match="every split"):
        assign_grouped_splits([row], proportions={Split.TRAIN: 1.0})
    with pytest.raises(ValidationError, match="sum to one"):
        assign_grouped_splits(
            [row],
            proportions={
                Split.TRAIN: 0.4,
                Split.VALIDATION: 0.2,
                Split.TEST: 0.2,
                Split.CONFIRMATION: 0.3,
            },
        )
    assert assign_splits([row], seed=3)[0].source_task_id == "s0"


def test_nested_fold_validation_error_paths() -> None:
    rows = [_record(index, f"s{index}", f"f{index}") for index in range(5)]
    assigned = assign_grouped_splits(rows, seed=2)
    with pytest.raises(ValidationError, match="at least two"):
        nested_group_development_folds(assigned, outer_folds=1, inner_folds=2)
    with pytest.raises(ValidationError, match="fewer development groups"):
        nested_group_development_folds(assigned, outer_folds=10, inner_folds=2)
    with pytest.raises(ValidationError, match="fewer outer-training groups"):
        nested_group_development_folds(assigned, outer_folds=2, inner_folds=10)
