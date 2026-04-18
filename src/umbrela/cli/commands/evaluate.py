from __future__ import annotations

import argparse
from typing import Any

from umbrela.cli.adapters import make_file_artifact
from umbrela.cli.commands.common import ensure_file_exists
from umbrela.cli.errors import INVALID_ARGS_EXIT_CODE, CLIError
from umbrela.cli.logging_utils import setup_logging
from umbrela.cli.operations import run_evaluate
from umbrela.cli.responses import CommandResponse


def run_evaluate_command(
    args: argparse.Namespace,
    *,
    run_evaluate_fn: Any = run_evaluate,
) -> CommandResponse:
    setup_logging(getattr(args, "log_level", 0), quiet=getattr(args, "quiet", False))
    if args.backend != "ensemble" and not args.model:
        raise CLIError(
            "evaluate requires --model unless --backend ensemble is used",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_model",
            command="evaluate",
        )
    if args.backend == "ensemble" and (not args.llm_judges or not args.model_names):
        raise CLIError(
            "ensemble evaluation requires --llm-judges and --model-names",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_ensemble_config",
            command="evaluate",
        )
    if args.result_file is not None:
        ensure_file_exists(
            args.result_file, command="evaluate", field_name="result_file"
        )
    if args.dry_run or args.validate_only:
        return CommandResponse(
            command="evaluate",
            mode="validate" if args.validate_only else "dry-run",
            inputs={
                "backend": args.backend,
                "qrel": args.qrel,
                "result_file": args.result_file,
            },
            resolved={"judge_cat": args.judge_cat},
        )
    result = run_evaluate_fn(args)
    artifacts = [make_file_artifact("evaluation-output", result.result_path)]
    seen_paths = {artifacts[0]["path"]}
    extra_artifact_index = 0
    for path in result.artifact_paths:
        artifact_path = make_file_artifact(f"artifact-{extra_artifact_index}", path)
        if artifact_path["path"] in seen_paths:
            continue
        seen_paths.add(artifact_path["path"])
        artifacts.append(artifact_path)
        extra_artifact_index += 1
    return CommandResponse(
        command="evaluate",
        inputs={
            "backend": args.backend,
            "qrel": args.qrel,
            "result_file": args.result_file,
        },
        resolved={"judge_cat": args.judge_cat, "num_sample": args.num_sample},
        artifacts=artifacts,
        metrics=result.metrics,
        warnings=[result.stdout] if result.stdout.strip() else [],
    )
