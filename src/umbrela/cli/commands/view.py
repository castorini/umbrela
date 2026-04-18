from __future__ import annotations

import argparse

from umbrela.cli.adapters import make_data_artifact
from umbrela.cli.errors import VALIDATION_EXIT_CODE, CLIError
from umbrela.cli.responses import CommandResponse
from umbrela.cli.view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
)


def run_view_command(args: argparse.Namespace) -> CommandResponse:
    try:
        records = load_records(args.path)
        artifact_type = detect_artifact_type(records, args.artifact_type)
    except ViewError as error:
        raise CLIError(
            str(error),
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="invalid_view_input",
            command="view",
            details={"path": args.path, "artifact_type": args.artifact_type},
        ) from error

    view_summary = build_view_summary(
        args.path,
        records,
        artifact_type,
        record_limit=args.records,
        show_prompts=args.show_prompts,
    )
    return CommandResponse(
        command="view",
        mode="inspect",
        inputs={"path": args.path},
        resolved={
            "artifact_type": artifact_type,
            "records": args.records,
            "show_prompts": args.show_prompts,
            "color": args.color,
        },
        artifacts=[make_data_artifact("view-summary", view_summary)],
        metrics=view_summary["summary"],
    )
