from __future__ import annotations

import importlib.metadata

from fastapi import FastAPI

from .routes import build_router
from .runtime import ServerConfig


def create_app(server_config: ServerConfig) -> FastAPI:
    app = FastAPI(
        title="umbrela",
        version=importlib.metadata.version("umbrela"),
    )
    app.include_router(build_router(server_config))
    return app
