from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from typing import Any

from umbrela.cli.adapters import make_data_artifact, serialize_direct_judgment
from umbrela.cli.introspection import validate_judge_payload
from umbrela.cli.normalize import (
    normalize_direct_judge_input,
    unwrap_direct_judge_payload,
)
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
    include_trace: bool = False
    redact_prompts: bool = False
    log_level: int = 0
    quiet: bool = False


_OVERRIDABLE_FIELDS = {
    "backend",
    "model",
    "prompt_file",
    "prompt_type",
    "few_shot_count",
    "execution_mode",
    "max_concurrency",
    "use_azure_openai",
    "use_openrouter",
    "reasoning_effort",
    "device",
    "include_reasoning",
    "include_trace",
    "redact_prompts",
}


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
        include_trace=config.include_trace,
        redact_prompts=config.redact_prompts,
        log_level=config.log_level,
        quiet=config.quiet,
        qrel="dl19-passage",
        output="json",
    )


def _extract_override_payload(payload: dict[str, Any]) -> dict[str, Any]:
    override_payload = payload.get("overrides", {})
    if not isinstance(override_payload, dict):
        raise ValueError("overrides must be an object when provided")
    unwrapped_payload = unwrap_direct_judge_payload(payload)
    unwrapped_override_payload = unwrapped_payload.get("overrides", {})
    if not isinstance(unwrapped_override_payload, dict):
        raise ValueError("overrides must be an object when provided")
    combined = dict(override_payload)
    combined.update(unwrapped_override_payload)
    unknown_keys = sorted(set(combined) - _OVERRIDABLE_FIELDS)
    if unknown_keys:
        raise ValueError(
            "unsupported judge override field(s): " + ", ".join(unknown_keys)
        )
    return combined


def _merge_config_with_payload(
    payload: dict[str, Any],
    *,
    config: ServerConfig,
) -> ServerConfig:
    overrides = _extract_override_payload(payload)
    if not overrides:
        return config
    effective_values = asdict(config)
    effective_values.update(overrides)
    effective_config = replace(config, **effective_values)
    if effective_config.backend == "ensemble":
        raise ValueError("ensemble backend is not supported by the judge serve API")
    if effective_config.use_azure_openai and effective_config.use_openrouter:
        raise ValueError(
            "use_azure_openai and use_openrouter cannot both be true in overrides"
        )
    if effective_config.backend != "gpt" and (
        effective_config.use_azure_openai
        or effective_config.use_openrouter
        or effective_config.reasoning_effort is not None
    ):
        raise ValueError(
            "provider overrides and reasoning_effort are only supported for the gpt backend"
        )
    if effective_config.backend not in {"hf", "os"} and "device" in overrides:
        raise ValueError("device override is only supported for hf and os backends")
    return effective_config


def execute_direct_judge(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    judge_runner: Any | None = None,
) -> CommandResponse:
    validation = validate_judge_payload(payload)
    normalized = normalize_direct_judge_input(payload)
    runner = judge_runner or run_judge_direct
    judgments = [
        serialize_direct_judgment(
            judgment,
            include_reasoning=args.include_reasoning,
            include_trace=getattr(args, "include_trace", False),
            redact_prompts=getattr(args, "redact_prompts", False),
        )
        for judgment in runner(normalized, args)
    ]
    return CommandResponse(
        command="judge",
        inputs={"mode": "direct", "backend": args.backend},
        resolved={
            "input_mode": "direct",
            "execution_mode": args.execution_mode,
            "backend": args.backend,
            "model": args.model,
            "prompt_type": args.prompt_type,
            "few_shot_count": args.few_shot_count,
        },
        validation=validation,
        artifacts=[make_data_artifact("judgments", judgments)],
    )


def run_judge_request(
    payload: dict[str, Any], *, config: ServerConfig
) -> CommandResponse:
    effective_config = _merge_config_with_payload(payload, config=config)
    return execute_direct_judge(payload, args=_base_args(effective_config))


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
