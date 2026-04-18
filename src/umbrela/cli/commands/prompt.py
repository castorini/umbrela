from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from umbrela.cli.adapters import make_data_artifact
from umbrela.cli.commands.common import ensure_file_exists, read_direct_payload
from umbrela.cli.errors import INVALID_ARGS_EXIT_CODE, VALIDATION_EXIT_CODE, CLIError
from umbrela.cli.normalize import normalize_direct_judge_input
from umbrela.cli.prompt_view import (
    build_prompt_template_view,
    build_rendered_prompt_view,
    list_prompt_templates,
)
from umbrela.cli.responses import CommandResponse
from umbrela.prompts import get_prompt_template
from umbrela.utils import qrel_utils


def run_prompt_command(
    args: argparse.Namespace,
    *,
    qrel_utils_module: Any = qrel_utils,
) -> CommandResponse:
    if args.prompt_command == "list":
        catalog = list_prompt_templates()
        return CommandResponse(
            command="prompt",
            mode="inspect",
            resolved={"prompt_command": "list"},
            artifacts=[make_data_artifact("prompt-catalog", catalog)],
        )
    if args.prompt_command == "show":
        template = get_prompt_template(
            prompt_file=args.prompt_file,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
        )
        view = build_prompt_template_view(
            template,
            prompt_file=args.prompt_file,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
        )
        return CommandResponse(
            command="prompt",
            mode="inspect",
            resolved={
                "prompt_command": "show",
                "prompt_file": view["selector"]["prompt_file"],
                "prompt_type": view["selector"]["prompt_type"],
                "few_shot_count": view["selector"]["few_shot_count"],
            },
            artifacts=[make_data_artifact("prompt-template", view)],
        )
    payload = read_direct_payload(args)
    normalized = normalize_direct_judge_input(payload)
    candidates = cast(list[dict[str, Any]], normalized["candidates"])
    if not 0 <= args.candidate_index < len(candidates):
        raise CLIError(
            "--candidate-index is out of range for the input payload",
            exit_code=VALIDATION_EXIT_CODE,
            status="validation_error",
            error_code="invalid_candidate_index",
            command="prompt",
            details={
                "candidate_index": args.candidate_index,
                "candidate_count": len(candidates),
            },
        )
    template = get_prompt_template(
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        few_shot_count=args.few_shot_count,
    )
    query = str(cast(dict[str, Any], normalized["query"])["text"])
    passage = str(candidates[args.candidate_index]["doc"]["segment"])
    if args.examples_file is not None:
        ensure_file_exists(
            args.examples_file, command="prompt", field_name="examples_file"
        )
        examples = Path(args.examples_file).read_text(encoding="utf-8")
    elif args.examples_text is not None:
        examples = args.examples_text
    elif args.few_shot_count > 0:
        if args.qrel is None:
            raise CLIError(
                "prompt render with --few-shot-count > 0 requires --qrel, "
                "--examples-text, or --examples-file",
                exit_code=INVALID_ARGS_EXIT_CODE,
                status="validation_error",
                error_code="missing_prompt_examples",
                command="prompt",
            )
        try:
            examples = qrel_utils_module.generate_examples_prompt(
                args.qrel, args.few_shot_count
            )
        except Exception as error:  # noqa: BLE001
            raise CLIError(
                f"Unable to generate prompt examples: {error}",
                exit_code=VALIDATION_EXIT_CODE,
                status="validation_error",
                error_code="prompt_example_generation_failed",
                command="prompt",
                details={"qrel": args.qrel, "few_shot_count": args.few_shot_count},
            ) from error
    else:
        examples = ""
    view = build_rendered_prompt_view(
        template,
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        few_shot_count=args.few_shot_count,
        candidate_index=args.candidate_index,
        qrel=args.qrel,
        query=query,
        passage=passage,
        examples=examples,
    )
    return CommandResponse(
        command="prompt",
        mode="inspect",
        inputs={"mode": "direct"},
        resolved={
            "prompt_command": "render",
            "prompt_file": view["selector"]["prompt_file"],
            "prompt_type": view["selector"]["prompt_type"],
            "few_shot_count": view["selector"]["few_shot_count"],
            "candidate_index": view["selector"]["candidate_index"],
            "qrel": args.qrel,
            "part": args.part,
        },
        artifacts=[make_data_artifact("rendered-prompt", view)],
    )
