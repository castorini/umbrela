from __future__ import annotations

import argparse
from typing import Any

from umbrela.api.runtime import execute_direct_judge
from umbrela.cli.adapters import make_data_artifact, make_file_artifact
from umbrela.cli.commands.common import (
    direct_judge_response_args,
    ensure_file_exists,
    filtered_records_from_judgments,
    prepare_output_path,
    read_direct_payload,
)
from umbrela.cli.errors import INVALID_ARGS_EXIT_CODE, VALIDATION_EXIT_CODE, CLIError
from umbrela.cli.introspection import validate_judge_batch_file
from umbrela.cli.io import read_jsonl, write_jsonl
from umbrela.cli.logging_utils import setup_logging
from umbrela.cli.normalize import prepare_direct_judge_payload
from umbrela.cli.operations import run_judge_batch, run_judge_direct
from umbrela.cli.responses import CommandResponse


def run_judge_command(
    args: argparse.Namespace,
    *,
    run_judge_batch_fn: Any = run_judge_batch,
    run_judge_direct_fn: Any = run_judge_direct,
) -> CommandResponse:
    setup_logging(getattr(args, "log_level", 0), quiet=getattr(args, "quiet", False))
    if args.execution_mode == "async" and args.backend != "gpt":
        raise CLIError(
            "--execution-mode async is currently supported only for --backend gpt",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="unsupported_execution_mode",
            command="judge",
        )
    if args.min_judgment is not None and not 0 <= args.min_judgment <= 3:
        raise CLIError(
            "--min-judgment must be between 0 and 3",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="invalid_min_judgment",
            command="judge",
        )
    if args.filtered_output_file is not None and args.min_judgment is None:
        raise CLIError(
            "--filtered-output-file requires --min-judgment",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_min_judgment",
            command="judge",
        )
    if args.input_file is not None:
        ensure_file_exists(args.input_file, command="judge", field_name="input_file")
        validation = validate_judge_batch_file(args.input_file)
        if not validation["valid"]:
            raise CLIError(
                "Batch judge input file does not match the expected request shape",
                exit_code=VALIDATION_EXIT_CODE,
                status="validation_error",
                error_code="invalid_input_file",
                command="judge",
            )
        records = read_jsonl(args.input_file)
        if args.dry_run:
            judgments: list[dict[str, Any]] = []
        else:
            judgments = run_judge_batch_fn(records, args)
        filtered_records: list[dict[str, Any]] = []
        if args.filtered_output_file is not None and not args.dry_run:
            filtered_records = filtered_records_from_judgments(
                records,
                judgments,
                min_judgment=args.min_judgment,
            )
        response = CommandResponse(
            command="judge",
            inputs={
                "mode": "batch",
                "input_file": args.input_file,
                "backend": args.backend,
            },
            resolved={
                "record_count": len(records),
                "execution_mode": args.execution_mode,
                "min_judgment": args.min_judgment,
            },
            validation=validation,
            artifacts=[make_data_artifact("judgments", judgments)],
        )
        if args.dry_run or args.validate_only:
            response.mode = "validate" if args.validate_only else "dry-run"
            response.artifacts = []
            return response
        if args.output_file is not None and not args.dry_run:
            output_path = prepare_output_path(args, command="judge")
            write_jsonl(output_path, judgments)
            response.artifacts.append(
                make_file_artifact("judgments-jsonl", output_path)
            )
        if args.filtered_output_file is not None and not args.dry_run:
            filtered_output_path = prepare_output_path(
                args,
                command="judge",
                attribute_name="filtered_output_file",
            )
            write_jsonl(filtered_output_path, filtered_records)
            response.artifacts.append(
                make_file_artifact("filtered-requests-jsonl", filtered_output_path)
            )
        return response

    payload = read_direct_payload(args)
    prepared = prepare_direct_judge_payload(payload)
    if args.dry_run or args.validate_only:
        return CommandResponse(
            command="judge",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"mode": "direct", "backend": args.backend},
            resolved={
                "input_mode": "direct",
                "execution_mode": args.execution_mode,
                "backend": args.backend,
                "model": args.model,
                "prompt_type": args.prompt_type,
                "few_shot_count": args.few_shot_count,
            },
            validation=prepared.validation,
            artifacts=[make_data_artifact("validated-request", prepared.normalized)],
        )
    return execute_direct_judge(
        payload,
        args=direct_judge_response_args(args),
        judge_runner=run_judge_direct_fn,
        prepared_payload=prepared,
    )
