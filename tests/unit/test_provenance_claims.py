from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_awareness_spectral.claims import (
    audit_claim_files,
    load_claim_ledger,
    packaged_results_schema,
    validate_claim_text,
    validate_result_manifest,
)
from eval_awareness_spectral.provenance import (
    assert_complete,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    hash_mapping,
    reserve_run_directory,
    sha256_file,
    validate_checksum,
)
from eval_awareness_spectral.schemas import (
    CacheConflictError,
    CompletenessRequirement,
    IncompleteRunError,
    ValidationError,
)


def _write_result_manifest(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset.jsonl"
    split = tmp_path / "split.jsonl"
    metric = tmp_path / "metric.json"
    dataset.write_text("{}\n", encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    metric.write_text('{"estimate": 1}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": "run-1",
        "dataset": {
            "location": dataset.name,
            "revision": "local-v1",
            "checksum": sha256_file(dataset),
        },
        "models": [{"model_id": "local/model", "revision": "rev", "family": "family"}],
        "tokenizer_revision": "tok-rev",
        "config_hash": "sha256:" + "1" * 64,
        "code_commit": "abcdef1",
        "environment_lock_hash": "sha256:" + "2" * 64,
        "seed": 42,
        "feature_definitions": ["feature"],
        "split_manifest": {
            "path": split.name,
            "checksum": sha256_file(split),
            "group_keys": ["source_task_id", "template_family"],
            "confirmation_locked": True,
        },
        "metrics": [
            {
                "claim_id": "claim-one",
                "definition": "synthetic metric",
                "split": "confirmation",
                "sample_size": 2,
                "result_path": metric.name,
                "checksum": sha256_file(metric),
            }
        ],
        "generated_at": "2026-07-22T12:00:00+02:00",
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_ledger(tmp_path: Path, claims: list[dict[str, object]]) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"claims": claims}), encoding="utf-8")
    return path


def test_hashing_atomic_writes_and_cache_identity(tmp_path: Path) -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert hash_mapping({"a": 2}).startswith("sha256:")
    raw = tmp_path / "raw.bin"
    atomic_write_bytes(raw, b"payload")
    assert raw.read_bytes() == b"payload"
    data = tmp_path / "value.json"
    atomic_write_json(data, {"b": 1, "a": 2})
    assert json.loads(data.read_text()) == {"a": 2, "b": 1}
    validate_checksum(raw, sha256_file(raw))
    with pytest.raises(IncompleteRunError, match="checksum mismatch"):
        validate_checksum(raw, "sha256:" + "0" * 64)

    run = reserve_run_directory(tmp_path / "run", "sha256:one")
    assert reserve_run_directory(run, "sha256:one") == run
    with pytest.raises(CacheConflictError, match="refusing"):
        reserve_run_directory(run, "sha256:two")


def test_completeness_accepts_crossed_values_and_rejects_missing() -> None:
    rows = [
        {"source_task_id": "a", "condition": "x", "model": "m", "metric": "z"},
        {"source_task_id": "b", "condition": "y", "model": "m", "metric": "z"},
    ]
    requirement = CompletenessRequirement(
        frozenset({"a", "b"}), frozenset({"x", "y"}), frozenset({"m"}), frozenset({"z"})
    )
    assert_complete(rows, requirement)
    assert_complete([], CompletenessRequirement())
    with pytest.raises(IncompleteRunError, match="missing source_task_id"):
        assert_complete(rows[:1], requirement)


def test_packaged_schema_and_valid_result_manifest(tmp_path: Path) -> None:
    assert packaged_results_schema()["title"].startswith("Eval-awareness")
    path = _write_result_manifest(tmp_path)
    assert validate_result_manifest(path)["run_id"] == "run-1"


def test_result_manifest_rejects_missing_version_path_and_checksum(tmp_path: Path) -> None:
    path = _write_result_manifest(tmp_path)
    value = json.loads(path.read_text())
    del value["run_id"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match="missing fields"):
        validate_result_manifest(path)

    path = _write_result_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["schema_version"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match="unsupported"):
        validate_result_manifest(path)

    path = _write_result_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["metrics"][0]["result_path"] = "absent.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match="does not exist"):
        validate_result_manifest(path)

    path = _write_result_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["metrics"][0]["checksum"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match="checksum mismatch"):
        validate_result_manifest(path)


def test_claim_ledger_validation_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    missing.write_text("{}")
    with pytest.raises(ValidationError, match="claims list"):
        load_claim_ledger(missing)

    duplicate = _write_ledger(
        tmp_path,
        [
            {"claim_id": "x", "status": "planned"},
            {"claim_id": "x", "status": "conjecture"},
        ],
    )
    with pytest.raises(ValidationError, match="duplicate"):
        load_claim_ledger(duplicate)

    invalid = _write_ledger(tmp_path, [{"claim_id": "x", "status": "unknown"}])
    with pytest.raises(ValidationError, match="invalid status"):
        load_claim_ledger(invalid)

    no_manifest = _write_ledger(tmp_path, [{"claim_id": "x", "status": "validated"}])
    with pytest.raises(ValidationError, match="lacks a result manifest"):
        load_claim_ledger(no_manifest)


def test_claim_text_requires_nearby_validated_marker() -> None:
    validate_claim_text("No result is asserted.", source="safe.md", validated_claim_ids=set())
    for phrase in (
        "Held-out AUC was 0.75.",
        "The probe achieved 0.94 AUROC on held-out tasks.",
        "The standardized effect was d = 0.8.",
    ):
        with pytest.raises(ValidationError, match="unmanifested empirical number"):
            validate_claim_text(phrase, source="bad.md", validated_claim_ids=set())
    validate_claim_text(
        "Held-out AUC was 0.75 (claim:claim-one).",
        source="good.md",
        validated_claim_ids={"claim-one"},
    )


def test_repository_claim_audit_resolves_validated_manifest(tmp_path: Path) -> None:
    manifest = _write_result_manifest(tmp_path)
    ledger = _write_ledger(
        tmp_path,
        [{"claim_id": "claim-one", "status": "validated", "manifest": manifest.name}],
    )
    prose = tmp_path / "README.md"
    prose.write_text("Held-out AUC was 0.75 (claim:claim-one).", encoding="utf-8")
    audit_claim_files([prose.name], ledger.name, repository_root=tmp_path)

    prose.write_text("Held-out AUC was 0.75.", encoding="utf-8")
    with pytest.raises(ValidationError, match="unmanifested"):
        audit_claim_files([prose.name], ledger.name, repository_root=tmp_path)
