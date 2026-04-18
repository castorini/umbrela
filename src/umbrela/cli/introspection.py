from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from umbrela.utils import qrel_utils


def doctor_report() -> dict[str, Any]:
    env_path = Path(".env")
    pyserini_available = qrel_utils.get_qrels_file is not None
    java_home = os.getenv("JAVA_HOME")
    python_ok = sys.version_info >= (3, 11)
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    openrouter_ready = bool(os.getenv("OPENROUTER_API_KEY"))
    azure_ready = bool(
        os.getenv("AZURE_OPENAI_API_BASE") and os.getenv("AZURE_OPENAI_API_VERSION")
    )
    gemini_ready = bool(os.getenv("GCLOUD_PROJECT") and os.getenv("GCLOUD_REGION"))
    hf_ready = bool(os.getenv("HF_TOKEN"))
    openai_dep_ready = importlib.util.find_spec("openai") is not None
    vertexai_dep_ready = importlib.util.find_spec("vertexai") is not None
    torch_dep_ready = importlib.util.find_spec("torch") is not None
    transformers_dep_ready = importlib.util.find_spec("transformers") is not None
    fastchat_dep_ready = importlib.util.find_spec("fastchat") is not None
    fastapi_dep_ready = importlib.util.find_spec("fastapi") is not None
    uvicorn_dep_ready = importlib.util.find_spec("uvicorn") is not None

    def status(
        *,
        ready: bool,
        missing_env: list[str] | None = None,
        missing_deps: list[str] | None = None,
    ) -> dict[str, Any]:
        missing_env = missing_env or []
        missing_deps = missing_deps or []
        if ready:
            state = "ready"
        elif missing_env:
            state = "missing_env"
        elif missing_deps:
            state = "missing_dependency"
        else:
            state = "blocked"
        return {
            "status": state,
            "missing_env": missing_env,
            "missing_dependencies": missing_deps,
        }

    backend_readiness = {
        "gpt": status(
            ready=python_ok
            and openai_dep_ready
            and (openai_ready or openrouter_ready or azure_ready),
            missing_env=[]
            if (openai_ready or openrouter_ready or azure_ready)
            else [
                "OPENAI_API_KEY or OPENROUTER_API_KEY or Azure OpenAI settings",
            ],
            missing_deps=[] if openai_dep_ready else ["openai"],
        ),
        "gemini": status(
            ready=python_ok and gemini_ready and vertexai_dep_ready,
            missing_env=[] if gemini_ready else ["GCLOUD_PROJECT", "GCLOUD_REGION"],
            missing_deps=[] if vertexai_dep_ready else ["vertexai"],
        ),
        "hf": status(
            ready=python_ok and hf_ready and torch_dep_ready and transformers_dep_ready,
            missing_env=[] if hf_ready else ["HF_TOKEN"],
            missing_deps=[
                dependency
                for dependency, available in (
                    ("torch", torch_dep_ready),
                    ("transformers", transformers_dep_ready),
                )
                if not available
            ],
        ),
        "os": status(
            ready=python_ok
            and torch_dep_ready
            and transformers_dep_ready
            and fastchat_dep_ready,
            missing_deps=[
                dependency
                for dependency, available in (
                    ("torch", torch_dep_ready),
                    ("transformers", transformers_dep_ready),
                    ("fastchat", fastchat_dep_ready),
                )
                if not available
            ],
        ),
        "ensemble": status(ready=python_ok),
    }
    command_readiness = {
        command: status(ready=python_ok)
        for command in [
            "judge",
            "evaluate",
            "view",
            "describe",
            "schema",
            "doctor",
            "validate",
        ]
    }
    command_readiness["serve"] = status(
        ready=python_ok and fastapi_dep_ready and uvicorn_dep_ready,
        missing_deps=[
            dependency
            for dependency, available in (
                ("fastapi", fastapi_dep_ready),
                ("uvicorn", uvicorn_dep_ready),
            )
            if not available
        ],
    )
    return {
        "python_version": sys.version.split()[0],
        "python_ok": python_ok,
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": openai_ready,
            "openrouter": openrouter_ready,
            "azure": azure_ready,
            "gemini": gemini_ready,
            "huggingface": hf_ready,
        },
        "pyserini_available": pyserini_available,
        "java_configured": bool(java_home),
        "optional_dependencies": {
            "fastapi": fastapi_dep_ready,
            "uvicorn": uvicorn_dep_ready,
        },
        "backend_readiness": backend_readiness,
        "command_readiness": command_readiness,
        "overall_status": "ready" if python_ok else "blocked",
    }


def validate_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from .normalize import normalize_direct_judge_input

    normalize_direct_judge_input(payload)
    return {"valid": True, "record_count": 1}


def validate_judge_batch_file(path: str) -> dict[str, Any]:
    from .io import read_jsonl

    records = read_jsonl(path)
    valid = True
    for record in records:
        query = record.get("query")
        candidates = record.get("candidates")
        if not (
            isinstance(query, dict)
            and isinstance(query.get("text"), str)
            and isinstance(candidates, list)
        ):
            valid = False
            break
    return {"valid": valid, "record_count": len(records)}
