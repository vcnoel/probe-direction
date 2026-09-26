"""Canonical hashes, atomic persistence, cache identity, and completeness checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .schemas import CacheConflictError, CompletenessRequirement, IncompleteRunError


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def hash_mapping(value: Mapping[str, Any]) -> str:
    return hash_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def reserve_run_directory(path: str | Path, config_hash: str) -> Path:
    """Create or validate a run directory without overwriting a different run."""

    run_dir = Path(path)
    identity = run_dir / "config.sha256"
    run_dir.mkdir(parents=True, exist_ok=True)
    if identity.exists():
        existing = identity.read_text(encoding="utf-8").strip()
        if existing != config_hash:
            raise CacheConflictError(
                f"output belongs to config {existing}; refusing config {config_hash}"
            )
    else:
        atomic_write_bytes(identity, f"{config_hash}\n".encode())
    return run_dir


def validate_checksum(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise IncompleteRunError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def assert_complete(rows: Iterable[Mapping[str, Any]], required: CompletenessRequirement) -> None:
    materialized = list(rows)
    dimensions = {
        "source_task_id": required.source_task_ids,
        "condition": required.conditions,
        "model": required.models,
        "metric": required.metrics,
    }
    for key, expected in dimensions.items():
        if not expected:
            continue
        observed = {str(row[key]) for row in materialized if key in row}
        missing = expected - observed
        if missing:
            raise IncompleteRunError(f"missing {key} values: {sorted(missing)}")
