"""Claim-ledger and result-manifest validation for release-safe prose."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from .provenance import sha256_file
from .schemas import ClaimStatus, ValidationError

_NUMBER = r"(?:\d+\.\d+|\.\d+|\d+)(?!\.\s)"
_METRIC = (
    r"(?:auc|auroc|accuracy|selectivity|effect\s+size|p\s*[- ]?value|"
    r"confidence\s+interval|sample\s+size)"
)
_EMPIRICAL_NUMBER = re.compile(
    rf"(?ix)(?:"
    rf"\b{_METRIC}\b[^\n]{{0,40}}?{_NUMBER}"
    rf"|{_NUMBER}[^\n]{{0,40}}?\b{_METRIC}\b"
    rf"|\b(?:n|p|d(?:_?z)?|cohen'?s?\s+d)\s*[:=]\s*{_NUMBER}"
    rf"|{_NUMBER}\s*%\b)"
)


def packaged_results_schema() -> dict[str, Any]:
    resource = files("eval_awareness_spectral.resources").joinpath("results_manifest.schema.json")
    return cast(dict[str, Any], json.loads(resource.read_text(encoding="utf-8")))


def load_claim_ledger(path: str | Path) -> dict[str, Any]:
    ledger = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(ledger.get("claims"), list):
        raise ValidationError("claim ledger must contain a claims list")
    valid = {status.value for status in ClaimStatus}
    ids: set[str] = set()
    for claim in ledger["claims"]:
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in ids:
            raise ValidationError(f"invalid or duplicate claim_id: {claim_id!r}")
        ids.add(claim_id)
        if claim.get("status") not in valid:
            raise ValidationError(f"invalid status for {claim_id}: {claim.get('status')!r}")
        if claim.get("status") == ClaimStatus.VALIDATED.value and not claim.get("manifest"):
            raise ValidationError(f"validated claim {claim_id} lacks a result manifest")
    return ledger


def validate_result_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate required fields and every artifact checksum without network access."""

    path = Path(manifest_path)
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    required = set(packaged_results_schema()["required"])
    missing = required - set(value)
    if missing:
        raise ValidationError(f"result manifest missing fields: {sorted(missing)}")
    if value.get("schema_version") != 1:
        raise ValidationError("unsupported result-manifest schema_version")
    base = path.parent
    references = [value["dataset"], value["split_manifest"], *value["metrics"]]
    for reference in references:
        item_path = Path(
            reference.get("result_path", reference.get("path", reference.get("location", "")))
        )
        if not item_path.is_absolute():
            item_path = base / item_path
        if not item_path.is_file():
            raise ValidationError(f"manifest artifact does not exist: {item_path}")
        if sha256_file(item_path) != reference.get("checksum"):
            raise ValidationError(f"manifest checksum mismatch: {item_path}")
    return value


def validate_claim_text(text: str, *, source: str, validated_claim_ids: set[str]) -> None:
    """Reject empirical numbers unless a nearby ``claim:<id>`` marker is validated."""

    # Enumerators are not measurements. Two forms reached the number-then-metric arm and were
    # reported as unmanifested empirical values: a markdown ordered-list marker at the head of a
    # line, and the digits of a labelled rule such as "**S1.** The primary statistic is max|AUROC|",
    # where the 1 belongs to the label S1 and the metric merely follows within the pattern's reach.
    skip = {m.start(1) for m in re.finditer(r"(?m)^[ \t]{0,3}(\d{1,3})[.)](?=[ \t*_])", text)}
    skip |= {m.start(1) for m in re.finditer(r"[A-Za-z](\d{1,3})[.)]", text)}
    # A number inside a code span is part of a DEFINITION, not a reported measurement: the rule
    # "the primary statistic is `max |AUROC - 0.5|`" states what is computed, and its 0.5 is the
    # chance level in the formula. Blanked to spaces rather than deleted so every offset above and
    # every excerpt below still refers to the real position in the text.
    scanned = re.sub(r"```.*?```|`[^`\n]*`",
                     lambda m: " " * len(m.group(0)), text, flags=re.S)
    for match in _EMPIRICAL_NUMBER.finditer(scanned):
        if match.start() in skip:
            continue
        window = text[max(0, match.start() - 160) : min(len(text), match.end() + 160)]
        markers = set(re.findall(r"claim:([a-z0-9-]+)", window, flags=re.IGNORECASE))
        if not markers & validated_claim_ids:
            excerpt = " ".join(match.group(0).split())
            raise ValidationError(f"unmanifested empirical number in {source}: {excerpt!r}")


def audit_claim_files(
    paths: list[str | Path], ledger_path: str | Path, *, repository_root: str | Path = "."
) -> None:
    root = Path(repository_root)
    ledger_file = Path(ledger_path)
    if not ledger_file.is_absolute():
        ledger_file = root / ledger_file
    ledger = load_claim_ledger(ledger_file)
    validated: set[str] = set()
    for claim in ledger["claims"]:
        if claim["status"] != ClaimStatus.VALIDATED.value:
            continue
        validate_result_manifest(root / claim["manifest"])
        validated.add(claim["claim_id"])
    for candidate in paths:
        path = root / candidate
        validate_claim_text(
            path.read_text(encoding="utf-8"), source=str(candidate), validated_claim_ids=validated
        )
