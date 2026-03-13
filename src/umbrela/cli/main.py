from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast

from .adapters import make_data_artifact, make_file_artifact
from .introspection import (
    COMMAND_DESCRIPTIONS,
    SCHEMAS,
    doctor_report,
    validate_judge_batch_file,
    validate_judge_payload,
)
from .io import read_jsonl, write_jsonl
from .normalize import normalize_direct_judge_input
from .operations import run_evaluate, run_judge_batch, run_judge_direct
from .responses import CommandResponse
from .view import (
    ViewError,
    build_view_summary,
    detect_artifact_type,
    load_records,
    render_view_summary,
)

INVALID_ARGS_EXIT_CODE = 2
MISSING_RESOURCE_EXIT_CODE = 4
VALIDATION_EXIT_CODE = 5
RUNTIME_EXIT_CODE = 6
KNOWN_COMMANDS = (
    "judge",
    "evaluate",
    "view",
    "describe",
    "schema",
    "doctor",
    "validate",
)
TOP_LEVEL_EXAMPLES = (
    (
        "umbrela judge --backend gpt --model gpt-4o "
        '--input-json \'{"query":"q","candidates":["p"]}\' --output json'
    ),
    (
        "umbrela evaluate --backend gpt --model gpt-4o "
        "--qrel dl19-passage --result-file run.trec --output json"
    ),
    "umbrela doctor --output json",
)


class CLIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        status: str,
        error_code: str,
        command: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.status = status
        self.error_code = error_code
        self.command = command or "unknown"
        self.details = details or {}


class CLIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        if message == "the following arguments are required: command":
            raise CLIError(
                _build_missing_command_message(),
                exit_code=INVALID_ARGS_EXIT_CODE,
                status="validation_error",
                error_code="missing_command",
                details={
                    "available_commands": list(KNOWN_COMMANDS),
                    "examples": list(TOP_LEVEL_EXAMPLES),
                    "help_hint": "Run `umbrela --help` for full usage.",
                },
            )
        raise CLIError(
            message,
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="invalid_arguments",
            command=_detect_command(sys.argv[1:]),
        )


def _detect_command(argv: Sequence[str]) -> str:
    for token in argv:
        if token in KNOWN_COMMANDS:
            return token
    return "unknown"


def _build_missing_command_message() -> str:
    command_list = ", ".join(KNOWN_COMMANDS)
    examples = "\n".join(f"  {example}" for example in TOP_LEVEL_EXAMPLES)
    return (
        "No command provided. Choose one of: "
        f"{command_list}\n"
        "Examples:\n"
        f"{examples}\n"
        "Run `umbrela --help` for full usage."
    )


def _emit_json(data: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(data) + "\n")


def _build_error_response(error: CLIError) -> CommandResponse:
    return CommandResponse(
        command=error.command,
        status=error.status,
        exit_code=error.exit_code,
        errors=[
            {
                "code": error.error_code,
                "message": error.message,
                "details": error.details,
                "retryable": False,
            }
        ],
    )


def _ensure_file_exists(path: str, *, command: str, field_name: str) -> None:
    if not Path(path).exists():
        raise CLIError(
            f"{field_name} does not exist: {path}",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_input",
            command=command,
            details={"field": field_name, "path": path},
        )


def _resolve_write_policy(args: argparse.Namespace) -> str:
    if getattr(args, "resume", False):
        return "resume"
    if getattr(args, "overwrite", False):
        return "overwrite"
    if getattr(args, "fail_if_exists", False):
        return "fail_if_exists"
    return "default_fail_if_exists"


def _prepare_output_path(args: argparse.Namespace, *, command: str) -> str:
    output_path = getattr(args, "output_file", None)
    if output_path is None:
        raise CLIError(
            f"{command} requires --output-file",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_output_file",
            command=command,
        )
    output_file = Path(cast(str, output_path))
    write_policy = _resolve_write_policy(args)
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


def _read_direct_payload(args: argparse.Namespace) -> dict[str, Any]:
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


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(prog="umbrela")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    judge_parser = subparsers.add_parser("judge")
    judge_inputs = judge_parser.add_mutually_exclusive_group(required=True)
    judge_inputs.add_argument("--input-file", type=str)
    judge_inputs.add_argument("--stdin", action="store_true")
    judge_inputs.add_argument("--input-json", type=str)
    judge_parser.add_argument(
        "--backend",
        choices=["gpt", "gemini", "hf", "os"],
        required=True,
    )
    judge_parser.add_argument("--model", type=str, required=True)
    judge_parser.add_argument("--output-file", type=str)
    judge_parser.add_argument(
        "--output", choices=["text", "json", "jsonl"], default="text"
    )
    judge_parser.add_argument("--prompt-file", type=str)
    judge_parser.add_argument("--prompt-type", type=str, default="bing")
    judge_parser.add_argument("--few-shot-count", type=int, default=0)
    judge_parser.add_argument(
        "--execution-mode", choices=["sync", "async"], default="sync"
    )
    judge_parser.add_argument("--max-concurrency", type=int, default=8)
    judge_parser.add_argument("--use-azure-openai", action="store_true")
    judge_parser.add_argument("--use-openrouter", action="store_true")
    judge_parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    judge_parser.add_argument("--device", type=str, default="cuda")
    judge_parser.add_argument("--include-reasoning", action="store_true")
    judge_parser.add_argument("--dry-run", action="store_true")
    judge_parser.add_argument("--overwrite", action="store_true")
    judge_parser.add_argument("--resume", action="store_true")
    judge_parser.add_argument("--fail-if-exists", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--backend", choices=["gpt", "gemini", "hf", "os", "ensemble"], required=True
    )
    evaluate_parser.add_argument("--model", type=str)
    evaluate_parser.add_argument("--qrel", type=str, required=True)
    evaluate_parser.add_argument("--result-file", type=str)
    evaluate_parser.add_argument("--prompt-file", type=str)
    evaluate_parser.add_argument("--prompt-type", type=str, default="bing")
    evaluate_parser.add_argument("--few-shot-count", type=int, default=0)
    evaluate_parser.add_argument("--num-sample", type=int, default=1)
    evaluate_parser.add_argument("--judge-cat", type=str, default="0,1,2,3")
    evaluate_parser.add_argument("--regenerate", action="store_true")
    evaluate_parser.add_argument("--output", choices=["text", "json"], default="text")
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.add_argument("--llm-judges", type=str)
    evaluate_parser.add_argument("--model-names", type=str)
    evaluate_parser.add_argument("--max-concurrency", type=int, default=8)
    evaluate_parser.add_argument("--use-azure-openai", action="store_true")
    evaluate_parser.add_argument("--use-openrouter", action="store_true")
    evaluate_parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"]
    )
    evaluate_parser.add_argument("--device", type=str, default="cuda")

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("target", choices=sorted(COMMAND_DESCRIPTIONS))
    describe_parser.add_argument("--output", choices=["text", "json"], default="text")

    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("target", choices=sorted(SCHEMAS))
    schema_parser.add_argument("--output", choices=["text", "json"], default="text")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")

    view_parser = subparsers.add_parser("view")
    view_parser.add_argument("path", type=str)
    view_parser.add_argument("--type", dest="artifact_type", type=str)
    view_parser.add_argument("--records", type=int, default=3)
    view_parser.add_argument(
        "--color", choices=["auto", "always", "never"], default="auto"
    )
    view_parser.add_argument("--show-prompts", action="store_true")
    view_parser.add_argument("--output", choices=["text", "json"], default="text")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("target", choices=["judge", "evaluate"])
    validate_inputs = validate_parser.add_mutually_exclusive_group()
    validate_inputs.add_argument("--input-file", type=str)
    validate_inputs.add_argument("--stdin", action="store_true")
    validate_inputs.add_argument("--input-json", type=str)
    validate_parser.add_argument("--qrel", type=str)
    validate_parser.add_argument("--result-file", type=str)
    validate_parser.add_argument("--output", choices=["text", "json"], default="text")

    return parser


def _format_text_judgments(
    judgments: list[dict[str, Any]], include_reasoning: bool
) -> str:
    lines: list[str] = []
    for index, judgment in enumerate(judgments, start=1):
        lines.append(
            f"[{index}] score={judgment['judgment']} status={judgment['result_status']}"
        )
        lines.append(f"query: {judgment['query']}")
        lines.append(f"passage: {judgment['passage']}")
        if include_reasoning and judgment.get("reasoning"):
            lines.append(f"reasoning: {judgment['reasoning']}")
    return "\n".join(lines)


def _run_judge_command(args: argparse.Namespace) -> CommandResponse:
    if args.execution_mode == "async" and args.backend != "gpt":
        raise CLIError(
            "--execution-mode async is currently supported only for --backend gpt",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="unsupported_execution_mode",
            command="judge",
        )
    if args.input_file is not None:
        _ensure_file_exists(args.input_file, command="judge", field_name="input_file")
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
            judgments = run_judge_batch(records, args)
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
            },
            validation=validation,
            artifacts=[make_data_artifact("judgments", judgments)],
        )
        if args.output_file is not None and not args.dry_run:
            output_path = _prepare_output_path(args, command="judge")
            write_jsonl(output_path, judgments)
            response.artifacts.append(
                make_file_artifact("judgments-jsonl", output_path)
            )
        return response

    payload = _read_direct_payload(args)
    validation = validate_judge_payload(payload)
    normalized = normalize_direct_judge_input(payload)
    judgments = [] if args.dry_run else run_judge_direct(normalized, args)
    if not args.include_reasoning:
        for judgment in judgments:
            judgment.pop("reasoning", None)
    return CommandResponse(
        command="judge",
        inputs={"mode": "direct", "backend": args.backend},
        resolved={
            "execution_mode": args.execution_mode,
            "normalized_request": normalized,
        },
        validation=validation,
        artifacts=[make_data_artifact("judgments", judgments)],
    )


def _run_evaluate_command(args: argparse.Namespace) -> CommandResponse:
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
        _ensure_file_exists(
            args.result_file, command="evaluate", field_name="result_file"
        )
    if args.dry_run:
        return CommandResponse(
            command="evaluate",
            mode="dry_run",
            inputs={
                "backend": args.backend,
                "qrel": args.qrel,
                "result_file": args.result_file,
            },
            resolved={"judge_cat": args.judge_cat},
        )
    result = run_evaluate(args)
    artifacts = [make_file_artifact("evaluation-output", result.result_path)]
    seen_paths = {artifacts[0]["path"]}
    extra_artifact_index = 0
    for path in result.artifact_paths:
        artifact_path = make_file_artifact(
            f"artifact-{extra_artifact_index}",
            path,
        )
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


def _run_describe_command(args: argparse.Namespace) -> CommandResponse:
    return CommandResponse(
        command="describe",
        mode="inspect",
        artifacts=[make_data_artifact(args.target, COMMAND_DESCRIPTIONS[args.target])],
    )


def _run_schema_command(args: argparse.Namespace) -> CommandResponse:
    return CommandResponse(
        command="schema",
        mode="inspect",
        artifacts=[make_data_artifact(args.target, SCHEMAS[args.target])],
    )


def _run_doctor_command() -> CommandResponse:
    return CommandResponse(command="doctor", mode="inspect", metrics=doctor_report())


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
    _ensure_file_exists(args.path, command="view", field_name="path")
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


def _run_validate_command(args: argparse.Namespace) -> CommandResponse:
    response = CommandResponse(command="validate", mode="inspect")
    if args.target == "judge":
        if args.input_file is not None:
            _ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            response.validation = validate_judge_batch_file(args.input_file)
        else:
            payload = _read_direct_payload(args)
            response.validation = validate_judge_payload(payload)
        return response
    if args.qrel is None:
        raise CLIError(
            "validate evaluate requires --qrel",
            exit_code=INVALID_ARGS_EXIT_CODE,
            status="validation_error",
            error_code="missing_qrel",
            command="validate",
        )
    if args.result_file is not None:
        _ensure_file_exists(
            args.result_file, command="validate", field_name="result_file"
        )
    response.validation = {
        "valid": True,
        "qrel": args.qrel,
        "result_file_present": args.result_file is not None,
    }
    return response


def _run_command(args: argparse.Namespace) -> CommandResponse:
    if args.command == "judge":
        return _run_judge_command(args)
    if args.command == "evaluate":
        return _run_evaluate_command(args)
    if args.command == "view":
        return _run_view_command(args)
    if args.command == "describe":
        return _run_describe_command(args)
    if args.command == "schema":
        return _run_schema_command(args)
    if args.command == "doctor":
        return _run_doctor_command()
    if args.command == "validate":
        return _run_validate_command(args)
    raise CLIError(
        f"Unknown command: {args.command}",
        exit_code=INVALID_ARGS_EXIT_CODE,
        status="validation_error",
        error_code="unknown_command",
        command=str(args.command),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        response = _run_command(args)
    except CLIError as error:
        response = _build_error_response(error)
        if getattr(error, "command", "unknown") == "unknown" and (
            argv is None or "--output" not in argv or "json" not in argv
        ):
            sys.stderr.write(error.message + "\n")
            return error.exit_code
        _emit_json(response.to_envelope())
        return error.exit_code
    except Exception as error:  # noqa: BLE001
        response = CommandResponse(
            command=getattr(locals().get("args", None), "command", "unknown"),
            status="runtime_error",
            exit_code=RUNTIME_EXIT_CODE,
            errors=[
                {
                    "code": "runtime_error",
                    "message": str(error),
                    "details": {},
                    "retryable": False,
                }
            ],
        )
        _emit_json(response.to_envelope())
        return RUNTIME_EXIT_CODE

    if getattr(args, "output", "text") == "json":
        _emit_json(response.to_envelope())
    elif getattr(args, "output", "text") == "jsonl":
        artifact = response.artifacts[0]["data"] if response.artifacts else []
        for record in artifact:
            sys.stdout.write(json.dumps(record) + "\n")
    elif args.command == "judge":
        artifact = cast(list[dict[str, Any]], response.artifacts[0]["data"])
        sys.stdout.write(
            _format_text_judgments(artifact, include_reasoning=args.include_reasoning)
            + "\n"
        )
    elif args.command in {"describe", "schema"}:
        sys.stdout.write(json.dumps(response.artifacts[0]["data"], indent=2) + "\n")
    elif args.command == "doctor":
        sys.stdout.write(json.dumps(response.metrics, indent=2) + "\n")
    elif args.command == "view":
        sys.stdout.write(
            render_view_summary(
                cast(dict[str, Any], response.artifacts[0]["data"]),
                color=args.color,
            )
            + "\n"
        )
    elif args.command == "validate":
        sys.stdout.write(json.dumps(response.validation, indent=2) + "\n")
    else:
        sys.stdout.write(json.dumps(response.to_envelope(), indent=2) + "\n")
    return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
