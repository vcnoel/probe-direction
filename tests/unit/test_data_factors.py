from __future__ import annotations

import json

import pytest

from eval_awareness_spectral.data import (
    EvalAwareBenchAdapter,
    factor_vector,
    load_evalawarebench,
    set_factor,
)
from eval_awareness_spectral.schemas import ValidationError


def test_factor_dimensions_toggle_independently() -> None:
    original = factor_vector({"monitoring": False, "stakes": "low"})
    changed = set_factor(original, "monitoring", True)

    assert {factor.name: factor.value for factor in changed} == {
        "monitoring": True,
        "stakes": "low",
    }
    assert original != changed


def test_local_adapter_preserves_distinct_endpoints(tmp_path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_task_id": "task-1",
                "recognition_label": "recognized",
                "behavioral_outcome": 0.25,
                "environment_factors": {"monitoring": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = EvalAwareBenchAdapter.load(path)[0]
    assert row["recognition_label"] == "recognized"
    assert row["behavioral_outcome"] == 0.25


@pytest.mark.parametrize("path", ["https://example.test/data.jsonl", "hf://dataset/data.csv"])
def test_adapter_rejects_nonlocal_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="local path"):
        load_evalawarebench(path)


def test_adapter_rejects_legacy_endpoint_alias(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text('{"label":"aware","factor_monitoring":true}\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="legacy alias"):
        load_evalawarebench(path)
