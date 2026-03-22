from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from umbrela.cli.adapters import make_data_artifact
from umbrela.cli.introspection import validate_judge_payload
from umbrela.cli.normalize import normalize_direct_judge_input
from umbrela.cli.operations import run_judge_direct
from umbrela.cli.responses import CommandResponse


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    backend: str
    model: str
    prompt_file: str | None = None
    prompt_type: str = "bing"
    few_shot_count: int = 0
    execution_mode: str = "sync"
    max_concurrency: int = 8
    use_azure_openai: bool = False
    use_openrouter: bool = False
    reasoning_effort: str | None = None
    device: str = "cuda"
    include_reasoning: bool = False
    log_level: int = 0
    quiet: bool = False


def _base_args(config: ServerConfig) -> argparse.Namespace:
    return argparse.Namespace(
        command="judge",
        backend=config.backend,
        model=config.model,
        prompt_file=config.prompt_file,
        prompt_type=config.prompt_type,
        few_shot_count=config.few_shot_count,
        execution_mode=config.execution_mode,
        max_concurrency=config.max_concurrency,
        use_azure_openai=config.use_azure_openai,
        use_openrouter=config.use_openrouter,
        reasoning_effort=config.reasoning_effort,
        device=config.device,
        include_reasoning=config.include_reasoning,
        log_level=config.log_level,
        quiet=config.quiet,
        qrel="dl19-passage",
        output="json",
    )


def execute_direct_judge(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    judge_runner: Any | None = None,
) -> CommandResponse:
    validation = validate_judge_payload(payload)
    normalized = normalize_direct_judge_input(payload)
    runner = judge_runner or run_judge_direct
    judgments = runner(normalized, args)
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


def run_judge_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    return execute_direct_judge(payload, args=_base_args(config))


def validation_error_response(message: str) -> CommandResponse:
    return CommandResponse(
        command="judge",
        status="validation_error",
        exit_code=5,
        errors=[
            {
                "code": "validation_error",
                "message": message,
                "details": {},
                "retryable": False,
            }
        ],
    )


def runtime_error_response(error: Exception) -> CommandResponse:
    return CommandResponse(
        command="judge",
        status="runtime_error",
        exit_code=6,
        errors=[
            {
                "code": "runtime_error",
                "message": str(error),
                "details": {},
                "retryable": False,
            }
        ],
    )
