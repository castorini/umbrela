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
from .logging_utils import setup_logging
from .normalize import normalize_direct_judge_input
from .operations import run_evaluate, run_judge_batch, run_judge_direct
from .responses import CommandResponse
from umbrela.utils import qrel_utils
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


def _wants_json(argv: Sequence[str]) -> bool:
    for index, token in enumerate(argv):
        if token == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
    return False


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


def _build_runtime_error_response(command: str, error: Exception) -> CommandResponse:
    return CommandResponse(
        command=command,
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


def _prepare_output_path(
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


def _write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n",
        encoding="utf-8",
    )


def _filtered_records_from_judgments(
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
            for candidate, judgment in zip(candidates, record_judgments)
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
    parser = CLIArgumentParser(
        prog="umbrela",
        description=(
            "Umbrela packaged CLI for direct judging, qrel-backed evaluation, "
            "validation, and artifact inspection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common patterns:\n"
            "  umbrela judge --backend gpt --model gpt-4o "
            '--input-json \'{"query":"q","candidates":["p"]}\' --output json\n'
            "  umbrela evaluate --backend gpt --model gpt-4o "
            "--qrel dl19-passage --result-file run.trec --output json\n"
            "  umbrela doctor --output json"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=CLIArgumentParser
    )

    judge_parser = subparsers.add_parser(
        "judge",
        help="Run a single judge backend over direct JSON input or batch JSONL input.",
        description=(
            "Run a single judge backend over direct JSON input or batch JSONL input."
        ),
    )
    judge_inputs = judge_parser.add_mutually_exclusive_group(required=True)
    judge_inputs.add_argument(
        "--input-file",
        type=str,
        help="Batch JSONL request file in the shared query-candidate schema.",
    )
    judge_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    judge_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON payload in the shared query-candidate schema.",
    )
    judge_parser.add_argument(
        "--backend",
        choices=["gpt", "gemini", "hf", "os"],
        required=True,
        help="Judge backend to execute.",
    )
    judge_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier for the selected backend.",
    )
    judge_parser.add_argument(
        "--output-file", type=str, help="Output JSONL path for batch judgments."
    )
    judge_parser.add_argument(
        "--output",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Human-readable text, machine-readable JSON envelope, or JSONL output.",
    )
    judge_parser.add_argument(
        "--prompt-file", type=str, help="Optional YAML prompt template override."
    )
    judge_parser.add_argument(
        "--prompt-type",
        type=str,
        default="bing",
        help="Built-in prompt template family.",
    )
    judge_parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Number of few-shot examples to inject into the prompt.",
    )
    judge_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode; async is currently supported only for the GPT backend.",
    )
    judge_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests for async GPT judging.",
    )
    judge_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for the GPT backend.",
    )
    judge_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for the GPT backend.",
    )
    judge_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for supported GPT-family models.",
    )
    judge_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device for local Hugging Face or FastChat backends.",
    )
    judge_parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include model reasoning fields in emitted results where available.",
    )
    judge_parser.add_argument(
        "--min-judgment",
        type=int,
        help="Minimum judgment threshold used with --filtered-output-file.",
    )
    judge_parser.add_argument(
        "--filtered-output-file",
        type=str,
        help=(
            "Write a filtered request JSONL containing only candidates "
            "meeting --min-judgment."
        ),
    )
    judge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs without running the judge backend.",
    )
    judge_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without running the judge backend.",
    )
    judge_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate an existing output file before writing judgments.",
    )
    judge_parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow writing to an existing output file without truncating it.",
    )
    judge_parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if the target output path already exists.",
    )
    judge_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )
    judge_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Generate modified qrels and evaluation metrics from a result file.",
        description=(
            "Generate modified qrels and evaluation metrics from a result file."
        ),
    )
    evaluate_parser.add_argument(
        "--backend",
        choices=["gpt", "gemini", "hf", "os", "ensemble"],
        required=True,
        help="Judge backend to execute.",
    )
    evaluate_parser.add_argument(
        "--model", type=str, help="Model identifier for the selected backend."
    )
    evaluate_parser.add_argument(
        "--qrel",
        type=str,
        required=True,
        help="Named qrel set to score against, such as dl19-passage.",
    )
    evaluate_parser.add_argument(
        "--result-file", type=str, help="Retrieval result file to evaluate."
    )
    evaluate_parser.add_argument(
        "--prompt-file", type=str, help="Optional YAML prompt template override."
    )
    evaluate_parser.add_argument(
        "--prompt-type",
        type=str,
        default="bing",
        help="Built-in prompt template family.",
    )
    evaluate_parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Number of few-shot examples to inject into the prompt.",
    )
    evaluate_parser.add_argument(
        "--num-sample", type=int, default=1, help="Number of judgment samples per pair."
    )
    evaluate_parser.add_argument(
        "--judge-cat",
        type=str,
        default="0,1,2,3",
        help="Comma-separated score categories used in evaluation.",
    )
    evaluate_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate modified qrels even if a cached one already exists.",
    )
    evaluate_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable summary or JSON envelope.",
    )
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate evaluation prerequisites without running judge backends.",
    )
    evaluate_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the declared contract without running judge backends.",
    )
    evaluate_parser.add_argument(
        "--llm-judges",
        type=str,
        help="Comma-separated judge backends for ensemble evaluation.",
    )
    evaluate_parser.add_argument(
        "--model-names",
        type=str,
        help="Comma-separated model names aligned with --llm-judges.",
    )
    evaluate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests for async GPT evaluation.",
    )
    evaluate_parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI environment settings for the GPT backend.",
    )
    evaluate_parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="Use OpenRouter for the GPT backend.",
    )
    evaluate_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for supported GPT-family models.",
    )
    evaluate_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device for local Hugging Face or FastChat backends.",
    )
    evaluate_parser.add_argument(
        "--manifest-path",
        type=str,
        help="Write the final JSON envelope to a manifest file.",
    )
    evaluate_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )

    describe_parser = subparsers.add_parser(
        "describe",
        help="Inspect structured metadata for a public Umbrela command.",
    )
    describe_parser.add_argument(
        "target",
        choices=sorted(COMMAND_DESCRIPTIONS),
        help="Public command to describe.",
    )
    describe_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable description or JSON envelope.",
    )

    schema_parser = subparsers.add_parser(
        "schema",
        help="Print JSON schemas for supported Umbrela inputs, outputs, and envelopes.",
    )
    schema_parser.add_argument(
        "target", choices=sorted(SCHEMAS), help="Schema artifact to print."
    )
    schema_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable schema or JSON envelope.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help=(
            "Report environment, dependency, and backend readiness for the "
            "packaged Umbrela CLI."
        ),
    )
    doctor_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable readiness report or JSON envelope.",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="Inspect an existing Umbrela artifact.",
        description=(
            "Inspect an existing Umbrela judgment artifact and render a stable summary."
        ),
    )
    view_parser.add_argument("path", type=str, help="Artifact path to inspect.")
    view_parser.add_argument(
        "--type",
        dest="artifact_type",
        type=str,
        help="Explicit artifact type when automatic detection is ambiguous.",
    )
    view_parser.add_argument(
        "--records",
        type=int,
        default=3,
        help="Number of records to sample in the inspection summary.",
    )
    view_parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color policy for text-mode rendering.",
    )
    view_parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Include sampled prompt text in the inspection summary.",
    )
    view_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable summary or JSON envelope.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help=(
            "Validate direct JSON input, batch JSONL input, or evaluation "
            "prerequisites without running models."
        ),
        description=(
            "Validate direct JSON input, batch JSONL input, or evaluation "
            "prerequisites without running models."
        ),
    )
    validate_parser.add_argument(
        "target", choices=["judge", "evaluate"], help="Validation target to inspect."
    )
    validate_inputs = validate_parser.add_mutually_exclusive_group()
    validate_inputs.add_argument(
        "--input-file", type=str, help="Batch JSONL request file to validate."
    )
    validate_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    validate_inputs.add_argument(
        "--input-json", type=str, help="Direct JSON payload to validate."
    )
    validate_parser.add_argument(
        "--qrel", type=str, help="Named qrel set required for evaluation validation."
    )
    validate_parser.add_argument(
        "--result-file",
        type=str,
        help="Retrieval result file required for evaluation validation.",
    )
    validate_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable validation summary or JSON envelope.",
    )

    return parser


def _format_text_judgments(
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


def _run_judge_command(args: argparse.Namespace) -> CommandResponse:
    setup_logging(getattr(args, "log_level", 0))
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
        filtered_records: list[dict[str, Any]] = []
        if args.filtered_output_file is not None and not args.dry_run:
            filtered_records = _filtered_records_from_judgments(
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
            output_path = _prepare_output_path(args, command="judge")
            write_jsonl(output_path, judgments)
            response.artifacts.append(
                make_file_artifact("judgments-jsonl", output_path)
            )
        if args.filtered_output_file is not None and not args.dry_run:
            filtered_output_path = _prepare_output_path(
                args,
                command="judge",
                attribute_name="filtered_output_file",
            )
            write_jsonl(filtered_output_path, filtered_records)
            response.artifacts.append(
                make_file_artifact(
                    "filtered-requests-jsonl",
                    filtered_output_path,
                )
            )
        return response

    payload = _read_direct_payload(args)
    validation = validate_judge_payload(payload)
    normalized = normalize_direct_judge_input(payload)
    if args.dry_run or args.validate_only:
        return CommandResponse(
            command="judge",
            mode="validate" if args.validate_only else "dry-run",
            inputs={"mode": "direct", "backend": args.backend},
            resolved={
                "execution_mode": args.execution_mode,
                "normalized_request": normalized,
            },
            validation=validation,
            artifacts=[make_data_artifact("validated-request", normalized)],
        )
    judgments = run_judge_direct(normalized, args)
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
    setup_logging(getattr(args, "log_level", 0))
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
    report = doctor_report()
    return CommandResponse(
        command="doctor",
        mode="inspect",
        metrics=report,
        validation={"python_ok": report["python_ok"]},
    )


def _run_view_command(args: argparse.Namespace) -> CommandResponse:
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
    response = CommandResponse(command="validate", mode="validate")
    if args.target == "judge":
        if args.input_file is not None:
            _ensure_file_exists(
                args.input_file, command="validate", field_name="input_file"
            )
            response.validation = validate_judge_batch_file(args.input_file)
        else:
            payload = _read_direct_payload(args)
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
    _ensure_file_exists(args.result_file, command="validate", field_name="result_file")
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
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    wants_json = _wants_json(argv)
    try:
        args = parser.parse_args(argv)
        response = _run_command(args)
    except CLIError as error:
        response = _build_error_response(error)
        if not wants_json:
            sys.stderr.write(error.message + "\n")
            return error.exit_code
        _emit_json(response.to_envelope())
        return error.exit_code
    except Exception as error:  # noqa: BLE001
        command = _detect_command(argv)
        response = _build_runtime_error_response(command, error)
        if wants_json:
            _emit_json(response.to_envelope())
        else:
            sys.stderr.write(f"{error}\n")
        return RUNTIME_EXIT_CODE

    _write_manifest(getattr(args, "manifest_path", None), response)

    if getattr(args, "output", "text") == "json":
        _emit_json(response.to_envelope())
    elif getattr(args, "output", "text") == "jsonl":
        artifact = response.artifacts[0]["data"] if response.artifacts else []
        for record in artifact:
            sys.stdout.write(json.dumps(record) + "\n")
    elif args.command == "judge":
        if not (
            getattr(args, "input_file", None) and getattr(args, "output_file", None)
        ):
            artifact = cast(list[dict[str, Any]], response.artifacts[0]["data"])
            sys.stdout.write(
                _format_text_judgments(
                    artifact, include_reasoning=args.include_reasoning
                )
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
