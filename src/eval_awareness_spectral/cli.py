"""CPU-safe release and provenance command line interface.

Importing or using any validation command never imports a model, tokenizer, dataset
client, or GPU stack. Activation extraction lives in scripts/ and pod/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .claims import audit_claim_files, packaged_results_schema, validate_result_manifest
from .config import load_config
from .data import read_manifest, validate_split_integrity
from .schemas import ProtocolError

LOGGER = logging.getLogger("eval_awareness_spectral")


class JsonFormatter(logging.Formatter):
    """Small structured formatter for machine-readable release checks."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname.lower(),
                "logger": record.name,
                "message": record.getMessage(),
            },
            sort_keys=True,
        )


def configure_logging(*, json_logs: bool = False, verbose: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
        force=True,
    )


def _write_or_print(value: Any, output: str | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        Path(output).write_text(rendered, encoding="utf-8")


def _command_schema(args: argparse.Namespace) -> None:
    _write_or_print(packaged_results_schema(), args.output)


def _command_validate_config(args: argparse.Namespace) -> None:
    config = load_config(args.path)
    print(json.dumps({"status": "valid", "config_hash": config.config_hash}, sort_keys=True))


def _command_validate_results(args: argparse.Namespace) -> None:
    value = validate_result_manifest(args.path)
    print(json.dumps({"status": "valid", "run_id": value["run_id"]}, sort_keys=True))


def _command_validate_prompts(args: argparse.Namespace) -> None:
    records = read_manifest(args.path)
    validate_split_integrity(records)
    print(json.dumps({"status": "valid", "rows": len(records)}, sort_keys=True))


def _command_claim_audit(args: argparse.Namespace) -> None:
    audit_claim_files(args.files, args.ledger, repository_root=args.root)
    print(json.dumps({"status": "valid", "files": len(args.files)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-awareness-spectral",
        description="Validate the evaluation-cue causal research protocol and provenance.",
    )
    parser.add_argument("--json-logs", action="store_true", help="emit structured JSON logs")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema", help="print the packaged results-manifest schema")
    schema.add_argument("--output", default=None)
    schema.set_defaults(handler=_command_schema)

    config = subparsers.add_parser("validate-config", help="validate and hash a protocol config")
    config.add_argument("path")
    config.set_defaults(handler=_command_validate_config)

    results = subparsers.add_parser(
        "validate-results", help="validate a result manifest and every checksum"
    )
    results.add_argument("path")
    results.set_defaults(handler=_command_validate_results)

    prompts = subparsers.add_parser(
        "validate-prompts", help="validate a local prompt manifest and split isolation"
    )
    prompts.add_argument("path")
    prompts.set_defaults(handler=_command_validate_prompts)

    claims = subparsers.add_parser(
        "claim-audit", help="reject empirical numbers lacking validated manifest claims"
    )
    claims.add_argument("--ledger", default="configs/claim_ledger.json")
    claims.add_argument("--root", default=".")
    claims.add_argument("files", nargs="+", help="repository-relative prose files")
    claims.set_defaults(handler=_command_claim_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    configure_logging(json_logs=args.json_logs, verbose=args.verbose)
    try:
        args.handler(args)
    except (ProtocolError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
