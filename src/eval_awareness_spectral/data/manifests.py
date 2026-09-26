"""Prompt-manifest construction and atomic JSONL persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..provenance import atomic_write_bytes, hash_bytes
from ..schemas import EnvironmentFactor, PromptRecord, Split, ValidationError
from .factors import FactorValue, factor_vector

_HASH_LENGTH = len("sha256:") + 64
_REQUIRED_FIELDS = {
    "row_id",
    "source_task_id",
    "template_family",
    "environment_factors",
    "condition_valence",
    "content_hash",
    "rendered_text_hash",
    "task_token_hash",
    "tokenizer_revision",
    "task_span_start",
    "task_span_end",
    "split",
}


def _digest_text(value: str) -> str:
    return hash_bytes(value.encode("utf-8"))


def stable_source_task_id(source: str, native_id: str) -> str:
    """Return an opaque ID stable across templates and generated variants."""

    if not source.strip() or not native_id.strip():
        raise ValidationError("source and native task ID must be non-empty")
    return _digest_text(f"{source}\0{native_id}")


def task_token_hash(token_ids: Sequence[int]) -> str:
    if not token_ids:
        raise ValidationError("task token sequence must be non-empty")
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise ValidationError("task token IDs must be non-negative integers")
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hash_bytes(payload)


def build_prompt_record(
    *,
    source_task_id: str,
    template_family: str,
    environment_factors: Mapping[str, FactorValue],
    condition_valence: str,
    content: str,
    rendered_text: str,
    task_tokens: Sequence[int],
    tokenizer_revision: str,
    task_span: tuple[int, int],
    split: Split | str,
    row_id: str | None = None,
    recognition_label: str | None = None,
    behavioral_outcome: float | str | None = None,
) -> PromptRecord:
    """Build one record while preserving factor, recognition, and outcome roles."""

    parsed_split = split if isinstance(split, Split) else Split(split)
    start, end = task_span
    if not source_task_id or not template_family or not condition_valence:
        raise ValidationError("source_task_id, template_family, and condition_valence are required")
    if not tokenizer_revision:
        raise ValidationError("tokenizer_revision is required")
    if start < 0 or end <= start or end - start != len(task_tokens):
        raise ValidationError("task span must be positive and match the task-token count")
    content_digest = _digest_text(content)
    rendered_digest = _digest_text(rendered_text)
    token_digest = task_token_hash(task_tokens)
    identity = row_id or _digest_text(
        "\0".join(
            [
                source_task_id,
                template_family,
                rendered_digest,
                tokenizer_revision,
                parsed_split.value,
            ]
        )
    )
    record = PromptRecord(
        row_id=identity,
        source_task_id=source_task_id,
        template_family=template_family,
        environment_factors=factor_vector(environment_factors),
        condition_valence=condition_valence,
        content_hash=content_digest,
        rendered_text_hash=rendered_digest,
        task_token_hash=token_digest,
        tokenizer_revision=tokenizer_revision,
        task_span_start=start,
        task_span_end=end,
        split=parsed_split,
        recognition_label=recognition_label,
        behavioral_outcome=behavioral_outcome,
    )
    validate_prompt_record(record)
    return record


def _validate_hash(name: str, value: str) -> None:
    if len(value) != _HASH_LENGTH or not value.startswith("sha256:"):
        raise ValidationError(f"{name} must be a sha256 digest")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValidationError(f"{name} must be a sha256 digest") from error


def validate_prompt_record(record: PromptRecord) -> None:
    if not record.row_id or not record.source_task_id or not record.template_family:
        raise ValidationError("record identities must be non-empty")
    if not record.environment_factors:
        raise ValidationError("environment factor vector must not be empty")
    if record.task_span_start < 0 or record.task_span_end <= record.task_span_start:
        raise ValidationError("invalid task span")
    for name in ("content_hash", "rendered_text_hash", "task_token_hash"):
        _validate_hash(name, str(getattr(record, name)))


def prompt_record_from_dict(value: Mapping[str, Any]) -> PromptRecord:
    missing = _REQUIRED_FIELDS - set(value)
    if missing:
        raise ValidationError(f"manifest row missing required fields: {sorted(missing)}")
    raw_factors = value["environment_factors"]
    if not isinstance(raw_factors, list):
        raise ValidationError("environment_factors must be a list")
    factors: list[EnvironmentFactor] = []
    for item in raw_factors:
        if not isinstance(item, Mapping) or set(item) != {"name", "value"}:
            raise ValidationError("each environment factor requires name and value")
        factors.append(EnvironmentFactor(name=str(item["name"]), value=item["value"]))
    try:
        record = PromptRecord(
            row_id=str(value["row_id"]),
            source_task_id=str(value["source_task_id"]),
            template_family=str(value["template_family"]),
            environment_factors=tuple(factors),
            condition_valence=str(value["condition_valence"]),
            content_hash=str(value["content_hash"]),
            rendered_text_hash=str(value["rendered_text_hash"]),
            task_token_hash=str(value["task_token_hash"]),
            tokenizer_revision=str(value["tokenizer_revision"]),
            task_span_start=int(value["task_span_start"]),
            task_span_end=int(value["task_span_end"]),
            split=Split(str(value["split"])),
            recognition_label=(
                None if value.get("recognition_label") is None else str(value["recognition_label"])
            ),
            behavioral_outcome=value.get("behavioral_outcome"),
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(f"invalid prompt-manifest row: {error}") from error
    validate_prompt_record(record)
    return record


def write_manifest(path: str | Path, records: Iterable[PromptRecord]) -> None:
    materialized = list(records)
    if not materialized:
        raise ValidationError("cannot write an empty prompt manifest")
    seen: set[str] = set()
    lines: list[str] = []
    for record in materialized:
        validate_prompt_record(record)
        if record.row_id in seen:
            raise ValidationError(f"duplicate row_id in prompt manifest: {record.row_id}")
        seen.add(record.row_id)
        lines.append(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def read_manifest(path: str | Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(value, Mapping):
                raise ValidationError(f"manifest line {line_number} must be an object")
            records.append(prompt_record_from_dict(value))
    if not records:
        raise ValidationError("prompt manifest contains no rows")
    if len({record.row_id for record in records}) != len(records):
        raise ValidationError("prompt manifest contains duplicate row IDs")
    return records
