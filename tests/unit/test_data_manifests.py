from __future__ import annotations

import json

import pytest

from eval_awareness_spectral.data import (
    build_prompt_record,
    read_manifest,
    stable_source_task_id,
    write_manifest,
)
from eval_awareness_spectral.schemas import Split, ValidationError


def test_manifest_round_trip_has_stable_identifiers_and_hashes(tmp_path) -> None:
    source_id = stable_source_task_id("synthetic", "native-1")
    record = build_prompt_record(
        source_task_id=source_id,
        template_family="direct-question",
        environment_factors={"monitoring": True, "stakes": "low"},
        condition_valence="evaluation",
        content="What is 2 + 2?",
        rendered_text="Evaluation: What is 2 + 2?",
        task_tokens=[101, 202, 303],
        tokenizer_revision="tokenizer@abc123",
        task_span=(2, 5),
        split=Split.TRAIN,
        recognition_label="aware",
        behavioral_outcome=1.0,
    )
    path = tmp_path / "manifest.jsonl"

    write_manifest(path, [record])
    loaded = read_manifest(path)

    assert loaded == [record]
    assert record.source_task_id == stable_source_task_id("synthetic", "native-1")
    assert record.content_hash.startswith("sha256:")
    assert record.rendered_text_hash != record.content_hash
    assert record.recognition_label == "aware"
    assert record.behavioral_outcome == 1.0


def test_manifest_rejects_tampered_hash(tmp_path) -> None:
    record = build_prompt_record(
        source_task_id="task-1",
        template_family="family-a",
        environment_factors={"cue": False},
        condition_valence="deployment",
        content="task",
        rendered_text="rendered task",
        task_tokens=[1],
        tokenizer_revision="tok-v1",
        task_span=(0, 1),
        split="test",
    )
    value = record.to_dict()
    value["content_hash"] = "sha256:not-a-hash"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="content_hash"):
        read_manifest(path)


def test_manifest_rejects_missing_required_field(tmp_path) -> None:
    path = tmp_path / "missing.jsonl"
    path.write_text('{"row_id":"x"}\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="missing required"):
        read_manifest(path)
