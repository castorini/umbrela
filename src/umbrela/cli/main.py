from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from umbrela.utils import qrel_utils

from .commands.common import format_text_judgments
from .commands.evaluate import run_evaluate_command
from .commands.judge import run_judge_command
from .commands.meta import run_describe_command, run_doctor_command, run_schema_command
from .commands.prompt import run_prompt_command
from .commands.serve import run_serve_command
from .commands.validate import run_validate_command
from .commands.view import run_view_command
from .config import load_config
from .errors import (
    INVALID_ARGS_EXIT_CODE,
    RUNTIME_EXIT_CODE,
    CLIError,
)
from .introspection import COMMAND_DESCRIPTIONS, SCHEMAS
from .operations import run_evaluate, run_judge_batch, run_judge_direct
from .prompt_view import (
    render_prompt_catalog_text,
    render_prompt_template_text,
    render_rendered_prompt_text,
)
from .responses import CommandResponse
from .view import render_view_summary

_shtab: Any | None
try:
    import shtab as _shtab
except ModuleNotFoundError:  # optional dev dependency
    _shtab = None

shtab = cast(Any, _shtab)

KNOWN_COMMANDS = (
    "judge",
    "evaluate",
    "serve",
    "view",
    "prompt",
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
    "umbrela serve --backend gpt --model gpt-4o --port 8084",
    "umbrela doctor --output json",
)


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


def _write_manifest(manifest_path: str | None, response: CommandResponse) -> None:
    if manifest_path is None:
        return
    Path(manifest_path).write_text(
        json.dumps(response.to_envelope(), indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> CLIArgumentParser:
    parser = CLIArgumentParser(
        prog="umbrela",
        description=(
            "umbrela packaged CLI for direct judging, qrel-backed evaluation, "
            "validation, and artifact inspection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common patterns:\n"
            "  umbrela judge --backend gpt --model gpt-4o "
            '--input-json \'{"query":"q","candidates":["p"]}\' --output json\n'
            "  umbrela serve --backend gpt --model gpt-4o --port 8084\n"
            "  umbrela evaluate --backend gpt --model gpt-4o "
            "--qrel dl19-passage --result-file run.trec --output json\n"
            "  umbrela prompt show --prompt-type bing --few-shot-count 0\n"
            "  umbrela doctor --output json"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('umbrela')}",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress all log output (sets log level to CRITICAL).",
    )
    if shtab is not None:
        shtab.add_argument_to(parser, ["--print-completion"])
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
        "--include-trace",
        action="store_true",
        help="Include prompt, raw prediction, and parse-status trace fields.",
    )
    judge_parser.add_argument(
        "--redact-prompts",
        action="store_true",
        help="Redact prompt text when --include-trace is enabled.",
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

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a FastAPI server for direct judge requests.",
        description=(
            "Start a FastAPI server that exposes direct umbrela judge requests "
            "over HTTP."
        ),
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8084)
    serve_parser.add_argument(
        "--backend",
        choices=["gpt", "gemini", "hf", "os"],
        required=True,
        help="Judge backend to execute.",
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier for the selected backend.",
    )
    serve_parser.add_argument("--prompt-file", type=str)
    serve_parser.add_argument("--prompt-type", type=str, default="bing")
    serve_parser.add_argument("--few-shot-count", type=int, default=0)
    serve_parser.add_argument(
        "--execution-mode",
        choices=["sync", "async"],
        default="sync",
        help="Execution mode; async is currently supported only for the GPT backend.",
    )
    serve_parser.add_argument("--max-concurrency", type=int, default=8)
    serve_parser.add_argument("--use-azure-openai", action="store_true")
    serve_parser.add_argument("--use-openrouter", action="store_true")
    serve_parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
    )
    serve_parser.add_argument("--device", type=str, default="cuda")
    serve_parser.add_argument("--include-reasoning", action="store_true")
    serve_parser.add_argument("--include-trace", action="store_true")
    serve_parser.add_argument("--redact-prompts", action="store_true")
    serve_parser.add_argument(
        "--log-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings, 1=info, 2=debug.",
    )

    describe_parser = subparsers.add_parser(
        "describe",
        help="Inspect structured metadata for a public umbrela command.",
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
        help="Print JSON schemas for supported umbrela inputs, outputs, and envelopes.",
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
            "packaged umbrela CLI."
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
        help="Inspect an existing umbrela artifact.",
        description=(
            "Inspect an existing umbrela judgment artifact and render a stable summary."
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

    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Inspect built-in or custom prompt templates.",
        description="Inspect built-in or custom prompt templates.",
    )
    prompt_subparsers = prompt_parser.add_subparsers(
        dest="prompt_command", required=True, parser_class=CLIArgumentParser
    )

    prompt_list_parser = prompt_subparsers.add_parser(
        "list",
        help="List built-in prompt templates.",
    )
    prompt_list_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable catalog or JSON envelope.",
    )

    prompt_show_parser = prompt_subparsers.add_parser(
        "show",
        help="Show a built-in or custom prompt template.",
    )
    prompt_show_source = prompt_show_parser.add_mutually_exclusive_group(required=True)
    prompt_show_source.add_argument(
        "--prompt-file", type=str, help="Custom YAML prompt template to inspect."
    )
    prompt_show_source.add_argument(
        "--prompt-type",
        choices=["basic", "bing"],
        help="Built-in prompt template family to inspect.",
    )
    prompt_show_parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Few-shot count used to resolve the built-in prompt template.",
    )
    prompt_show_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable template or JSON envelope.",
    )

    prompt_render_parser = prompt_subparsers.add_parser(
        "render",
        help="Render a built-in or custom prompt template against direct input.",
    )
    prompt_render_source = prompt_render_parser.add_mutually_exclusive_group(
        required=True
    )
    prompt_render_source.add_argument(
        "--prompt-file", type=str, help="Custom YAML prompt template to render."
    )
    prompt_render_source.add_argument(
        "--prompt-type",
        choices=["basic", "bing"],
        help="Built-in prompt template family to render.",
    )
    prompt_render_inputs = prompt_render_parser.add_mutually_exclusive_group(
        required=True
    )
    prompt_render_inputs.add_argument(
        "--stdin",
        action="store_true",
        help="Read one direct JSON payload from standard input.",
    )
    prompt_render_inputs.add_argument(
        "--input-json",
        type=str,
        help="Direct JSON payload in the shared query-candidate schema.",
    )
    prompt_render_parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Few-shot count used to resolve the prompt template.",
    )
    prompt_render_parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="Candidate index to render from the direct input payload.",
    )
    prompt_render_parser.add_argument(
        "--qrel",
        type=str,
        help="Named qrel used to generate few-shot examples when --few-shot-count > 0.",
    )
    prompt_render_examples = prompt_render_parser.add_mutually_exclusive_group()
    prompt_render_examples.add_argument(
        "--examples-text",
        type=str,
        help="Explicit example block to inject into the rendered prompt.",
    )
    prompt_render_examples.add_argument(
        "--examples-file",
        type=str,
        help=(
            "File containing the exact example block to inject into the rendered "
            "prompt."
        ),
    )
    prompt_render_parser.add_argument(
        "--part",
        choices=["system", "user", "all"],
        default="all",
        help="Rendered prompt section to show in text mode.",
    )
    prompt_render_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Human-readable rendered prompt or JSON envelope.",
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


def _run_command(args: argparse.Namespace) -> CommandResponse:
    if args.command == "judge":
        return run_judge_command(
            args,
            run_judge_batch_fn=run_judge_batch,
            run_judge_direct_fn=run_judge_direct,
        )
    if args.command == "evaluate":
        return run_evaluate_command(args, run_evaluate_fn=run_evaluate)
    if args.command == "serve":
        return run_serve_command(args)
    if args.command == "view":
        return run_view_command(args)
    if args.command == "prompt":
        return run_prompt_command(args, qrel_utils_module=qrel_utils)
    if args.command == "describe":
        return run_describe_command(args.target)
    if args.command == "schema":
        return run_schema_command(args.target)
    if args.command == "doctor":
        return run_doctor_command(config_path=getattr(args, "_config_path", None))
    if args.command == "validate":
        return run_validate_command(args)
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
    config, config_path = load_config()
    wants_json = _wants_json(argv)
    try:
        args = parser.parse_args(argv)
        args._config_path = config_path
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if flag not in argv:
                setattr(args, key, value)
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
                format_text_judgments(
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
    elif args.command == "prompt":
        if args.prompt_command == "list":
            sys.stdout.write(
                render_prompt_catalog_text(
                    cast(list[dict[str, Any]], response.artifacts[0]["data"])
                )
                + "\n"
            )
        elif args.prompt_command == "show":
            sys.stdout.write(
                render_prompt_template_text(
                    cast(dict[str, Any], response.artifacts[0]["data"])
                )
                + "\n"
            )
        else:
            sys.stdout.write(
                render_rendered_prompt_text(
                    cast(dict[str, Any], response.artifacts[0]["data"]),
                    part=args.part,
                )
                + "\n"
            )
    elif args.command == "validate":
        sys.stdout.write(json.dumps(response.validation, indent=2) + "\n")
    elif args.command == "serve":
        pass
    else:
        sys.stdout.write(json.dumps(response.to_envelope(), indent=2) + "\n")
    return response.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
