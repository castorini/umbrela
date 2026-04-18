from __future__ import annotations

import argparse

from umbrela.cli.commands.common import ensure_file_exists, read_direct_payload
from umbrela.cli.errors import INVALID_ARGS_EXIT_CODE, VALIDATION_EXIT_CODE, CLIError
from umbrela.cli.introspection import validate_judge_batch_file, validate_judge_payload
from umbrela.cli.responses import CommandResponse
from umbrela.utils import qrel_utils


def run_validate_command(args: argparse.Namespace) -> CommandResponse:
    response = CommandResponse(command="validate", mode="validate")
    if args.target == "judge":
        if args.input_file is not None:
            ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            response.validation = validate_judge_batch_file(args.input_file)
        else:
            payload = read_direct_payload(args)
            response.validation = validate_judge_payload(payload)
        response.status = (
            "success" if response.validation.get("valid", False) else "validation_error"
        )
        response.exit_code = 0 if response.status == "success" else VALIDATION_EXIT_CODE
        if response.status != "success":
            response.errors.append(
                {
                    "code": "validation_failed",
                    "message": "judge input failed validation",
                    "details": response.validation,
                    "retryable": False,
                }
            )
        return response
    if args.qrel is None:
        raise CLIError(
            "validate evaluate requires --qrel",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_qrel",
            command="validate",
        )
    if args.result_file is None:
        raise CLIError(
            "validate evaluate requires --result-file",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_result_file",
            command="validate",
        )
    ensure_file_exists(args.result_file, command="validate", field_name="result_file")
    qrel_supported = qrel_utils.get_qrels_file(args.qrel) is not None
    response.validation = {
        "valid": qrel_supported,
        "qrel": args.qrel,
        "result_file_present": True,
        "qrel_supported": qrel_supported,
    }
    response.status = "success" if qrel_supported else "validation_error"
    response.exit_code = 0 if qrel_supported else VALIDATION_EXIT_CODE
    if not qrel_supported:
        response.errors.append(
            {
                "code": "unsupported_qrel",
                "message": f"Unsupported qrel: {args.qrel}",
                "details": {"qrel": args.qrel},
                "retryable": False,
            }
        )
    return response
