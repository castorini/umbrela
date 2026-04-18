from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from umbrela.cli.errors import (
    INVALID_ARGS_EXIT_CODE,
    MISSING_RESOURCE_EXIT_CODE,
    RUNTIME_EXIT_CODE,
    VALIDATION_EXIT_CODE,
    CLIError,
)


def ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def prepare_output_path(
    args: argparse.Namespace,
    *,
    command: str,
    attribute_name: str = "output_file",
) -> str:
    output_path = getattr(args, attribute_name, None)
    if output_path is None:
        raise CLIError(
            f"{command} requires --{attribute_name.replace('_', '-')}",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code=f"missing_{attribute_name}",
            command=command,
        )
    output_file = Path(cast(str, output_path))
    write_policy = resolve_write_policy(args)
    if output_file.exists():
        if write_policy == "resume":
            return str(output_file)
        if write_policy == "overwrite":
            output_file.write_text("", encoding="utf-8")
            return str(output_file)
        raise CLIError(
            f"Output file already exists: {output_file}",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="write_policy_conflict",
            command=command,
            details={"path": str(output_file), "write_policy": write_policy},
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return str(output_file)


def filtered_records_from_judgments(
    records: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    min_judgment: int,
) -> list[dict[str, Any]]:
    filtered_records: list[dict[str, Any]] = []
    judgment_index = 0
    for record_index, record in enumerate(records, start=1):
        candidates = cast(list[dict[str, Any]], record["candidates"])
        next_index = judgment_index + len(candidates)
        record_judgments = judgments[judgment_index:next_index]
        if len(record_judgments) != len(candidates):
            raise CLIError(
                "judge output count does not match the number of input candidates",
                exit_code=RUNTIME_EXIT_CODE,
                status="runtime_error",
                error_code="judgment_count_mismatch",
                command="judge",
                details={
                    "record_index": record_index,
                    "candidate_count": len(candidates),
                    "judgment_count": len(record_judgments),
                },
            )
        kept_candidates = [
            candidate
            for candidate, judgment in zip(candidates, record_judgments, strict=True)
            if int(judgment["judgment"]) >= min_judgment
        ]
        if not kept_candidates:
            raise CLIError(
                "Filtering removed every candidate from a request",
                exit_code=RUNTIME_EXIT_CODE,
                status="runtime_error",
                error_code="empty_filtered_request",
                command="judge",
                details={
                    "record_index": record_index,
                    "min_judgment": min_judgment,
                    "query": record["query"],
                },
            )
        filtered_records.append(
            {
                "query": record["query"],
                "candidates": kept_candidates,
            }
        )
        judgment_index = next_index
    if judgment_index != len(judgments):
        raise CLIError(
            "judge output count does not match the input request file",
            exit_code=RUNTIME_EXIT_CODE,
            status="runtime_error",
            error_code="judgment_count_mismatch",
            command="judge",
            details={
                "consumed_judgments": judgment_index,
                "total_judgments": len(judgments),
            },
        )
    return filtered_records


def read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.stdin:
            return cast(dict[str, Any], json.loads(sys.stdin.read()))
        if args.input_json is not None:
            return cast(dict[str, Any], json.loads(args.input_json))
    except json.JSONDecodeError as exc:
        raise CLIError(
            "Input payload is not valid JSON",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_json",
            command=args.command,
            details={"error": str(exc)},
        ) from exc
    raise CLIError(
        "Direct input requires --stdin or --input-json",
        exit_code=INVALID_ARGS_EXIT_CODE,
        status="validation_error",
        error_code="missing_direct_input",
        command=args.command,
    )


def direct_judge_response_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "output", "text") != "text" or getattr(
        args, "include_trace", False
    ):
        return args
    response_args = deepcopy(args)
    response_args.include_trace = True
    return response_args


def format_text_judgments(
    judgments: list[dict[str, Any]], include_reasoning: bool
) -> str:
    blocks: list[str] = []
    for judgment in judgments:
        lines = [f"query: {judgment['query']}"]
        lines.append(f"candidate: {judgment['passage']}")
        lines.append(f"judgment: {judgment['judgment']}")
        if int(judgment.get("result_status", 1)) == 0:
            lines.append("parsing failed")
        if include_reasoning and judgment.get("reasoning"):
            lines.append(f"reasoning: {judgment['reasoning']}")
        blocks.append("\n".join(lines))
    return "\n-----\n".join(blocks)
