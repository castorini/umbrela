from __future__ import annotations

import argparse

from umbrela.api.runtime import ServerConfig
from umbrela.cli.errors import MISSING_RESOURCE_EXIT_CODE, CLIError
from umbrela.cli.responses import CommandResponse


def run_serve_command(args: argparse.Namespace) -> CommandResponse:
    try:
        import uvicorn

        from umbrela.api.app import create_app
    except ModuleNotFoundError as error:
        raise CLIError(
            "serve requires FastAPI dependencies; install the `api` extra",
            exit_code=MISSING_RESOURCE_EXIT_CODE,
            status="validation_error",
            error_code="missing_api_dependencies",
            command="serve",
            details={"missing_dependencies": ["fastapi", "uvicorn"]},
        ) from error

    app = create_app(
        ServerConfig(
            host=args.host,
            port=args.port,
            backend=args.backend,
            model=args.model,
            prompt_file=args.prompt_file,
            prompt_type=args.prompt_type,
            few_shot_count=args.few_shot_count,
            execution_mode=args.execution_mode,
            max_concurrency=args.max_concurrency,
            use_azure_openai=args.use_azure_openai,
            use_openrouter=args.use_openrouter,
            reasoning_effort=args.reasoning_effort,
            device=args.device,
            include_reasoning=args.include_reasoning,
            include_trace=args.include_trace,
            redact_prompts=args.redact_prompts,
            log_level=args.log_level,
            quiet=getattr(args, "quiet", False),
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return CommandResponse(
        command="serve", resolved={"host": args.host, "port": args.port}
    )
