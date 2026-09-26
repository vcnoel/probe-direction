import numpy as np
import pytest

from eval_awareness_spectral.evaluation.selectivity import (
    paired_decodability_selectivity,
    paired_selectivity,
)
from eval_awareness_spectral.evaluation.statistics import (
    above_chance_magnitude,
    orientation_invariant_decodability,
)
from eval_awareness_spectral.schemas import ValidationError


def test_reversed_scores_have_identical_decodability() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    assert orientation_invariant_decodability(labels, scores) == 1.0
    assert orientation_invariant_decodability(labels, -scores) == 1.0
    assert above_chance_magnitude(labels, -scores) == 0.5


def test_statistics_reject_non_finite_input() -> None:
    with pytest.raises(ValidationError, match="finite"):
        orientation_invariant_decodability([0, 1], [0.0, np.nan])


def test_selectivity_interaction_is_paired_by_source_task() -> None:
    labels = np.tile([0, 0, 1, 1], 2)
    tasks = np.repeat(["task-a", "task-b"], 4)
    # Target is perfect on A and chance on B; control reverses that pattern.
    target = np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=float)
    control = np.array([0, 2, 1, 3, 0, 1, 2, 3], dtype=float)
    result = paired_selectivity(labels, target, control, tasks, n_bootstrap=100, seed=2)
    assert result.task_ids.tolist() == ["task-a", "task-b"]
    assert result.paired_differences[0] == -result.paired_differences[1]
    assert result.estimate == pytest.approx(0.0)

    direct = paired_decodability_selectivity(
        [1.0, 0.75], [0.75, 1.0], ["task-a", "task-b"], n_bootstrap=100, seed=2
    )
    assert direct.estimate == pytest.approx(0.0)
