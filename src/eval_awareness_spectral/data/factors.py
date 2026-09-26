"""Environment-factor construction and local EvalAwareBench-style ingestion."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..schemas import EnvironmentFactor, ValidationError

FactorValue = str | bool | int | float

# These names collapsed a measured recognition target and an observed behavior in
# early data files.  Silently accepting them would make the two endpoints
# scientifically indistinguishable.
LEGACY_ALIASES = frozenset(
    {
        "label",
        "outcome",
        "evaluation_label",
        "eval_awareness_label",
        "recognition_outcome",
    }
)


def factor_vector(factors: Mapping[str, FactorValue]) -> tuple[EnvironmentFactor, ...]:
    """Return a canonical vector whose dimensions can be changed independently."""

    if not factors:
        raise ValidationError("environment factor vector must not be empty")
    vector: list[EnvironmentFactor] = []
    for name, value in sorted(factors.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("environment factor names must be non-empty strings")
        if name in LEGACY_ALIASES:
            raise ValidationError(f"legacy aliased field is prohibited: {name}")
        if isinstance(value, (dict, list, tuple)) or value is None:
            raise ValidationError(f"invalid value for environment factor {name!r}")
        vector.append(EnvironmentFactor(name=name, value=value))
    return tuple(vector)


def set_factor(
    factors: Iterable[EnvironmentFactor], name: str, value: FactorValue
) -> tuple[EnvironmentFactor, ...]:
    """Set exactly one factor, preserving every other factor dimension."""

    current = {factor.name: factor.value for factor in factors}
    current[name] = value
    return factor_vector(current)


def _require_local_data_path(path: str | Path) -> Path:
    text = str(path)
    parsed = urlparse(text)
    # A single-character scheme is a Windows drive letter, not a URL: urlparse("C:/data.jsonl")
    # reports scheme "c", so this guard rejected every absolute path on Windows and the adapter
    # could not load a local file there at all. CI runs Linux, where the bug is invisible.
    remote = (len(parsed.scheme) > 1 and parsed.scheme != "file") or bool(parsed.netloc)
    if remote:
        raise ValidationError("adapter accepts a caller-supplied local path only")
    local = Path(path)
    if not local.is_file():
        raise ValidationError(f"local dataset path does not exist or is not a file: {local}")
    if local.suffix.lower() not in {".jsonl", ".csv"}:
        raise ValidationError("adapter supports only local .jsonl and .csv files")
    return local


def _decode_csv_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in '[{"' or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _validate_adapter_row(row: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    prohibited = LEGACY_ALIASES.intersection(row)
    if prohibited:
        names = ", ".join(sorted(prohibited))
        raise ValidationError(f"row {line_number} uses prohibited legacy alias(es): {names}")
    if "recognition_label" not in row and "behavioral_outcome" not in row:
        raise ValidationError(
            f"row {line_number} must declare recognition_label and/or behavioral_outcome explicitly"
        )
    result = dict(row)
    raw_factors = result.get("environment_factors")
    if raw_factors is None:
        prefixed = {
            key.removeprefix("factor_"): value
            for key, value in result.items()
            if key.startswith("factor_")
        }
        if prefixed:
            raw_factors = prefixed
    if raw_factors is not None:
        if not isinstance(raw_factors, Mapping):
            raise ValidationError(f"row {line_number} environment_factors must be an object")
        result["environment_factors"] = {
            item.name: item.value for item in factor_vector(raw_factors)
        }
    return result


def load_evalawarebench(path: str | Path) -> list[dict[str, Any]]:
    """Load an EvalAwareBench-style file without downloads or implicit discovery.

    The adapter deliberately takes only a local JSONL/CSV path.  Endpoint fields
    retain their explicit ``recognition_label`` and ``behavioral_outcome`` names.
    """

    local = _require_local_data_path(path)
    rows: list[dict[str, Any]] = []
    if local.suffix.lower() == ".jsonl":
        with local.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValidationError(
                        f"invalid JSON on line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise ValidationError(f"row {line_number} must be a JSON object")
                rows.append(_validate_adapter_row(value, line_number))
    else:
        with local.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValidationError("CSV file must include a header")
            for line_number, row in enumerate(reader, 2):
                decoded = {key: _decode_csv_cell(value) for key, value in row.items()}
                rows.append(_validate_adapter_row(decoded, line_number))
    if not rows:
        raise ValidationError("dataset contains no rows")
    return rows


class EvalAwareBenchAdapter:
    """Namespace adapter for explicit local EvalAwareBench-style files."""

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        return load_evalawarebench(path)
