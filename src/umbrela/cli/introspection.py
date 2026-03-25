from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from umbrela.utils import qrel_utils

COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "judge": {
        "summary": (
            "Run a single umbrela backend over direct JSON input "
            "or batch JSONL requests."
        ),
        "execution_mode_default": "sync",
        "inspection_safe": False,
        "backends": ["gpt", "gemini", "hf", "os"],
        "examples": [
            (
                "umbrela judge --backend gpt --model gpt-4o "
                '--input-json \'{"query":"how long is life cycle of flea",'
                '"candidates":["The life cycle of a flea can last anywhere '
                "from 20 days to an entire year.\"]}' "
                "--output json"
            ),
            (
                "umbrela judge --backend gemini --model gemini-1.5-pro "
                "--input-file requests.jsonl --output-file judgments.jsonl"
            ),
            (
                "umbrela judge --backend gpt --model gpt-4o "
                "--input-file requests.jsonl --output-file judgments.jsonl "
                "--filtered-output-file relevant.jsonl --min-judgment 2"
            ),
            (
                "umbrela judge --backend gpt --model gpt-4o "
                '--input-json \'{"query":"q","candidates":["p"]}\' '
                "--include-trace --redact-prompts --output json"
            ),
        ],
        "direct_input": {
            "ids_optional": True,
            "shape": {
                "query": "string | {text: string, qid?: string}",
                "candidates": [
                    "string | {text: string, docid?: string} | "
                    "{doc: string | {segment: string} | "
                    "{contents: string}, docid?: string}"
                ],
            },
        },
        "batch_input": "JSONL records with query.text and candidates[].doc.segment.",
    },
    "evaluate": {
        "summary": (
            "Generate modified qrels and evaluation metrics "
            "using a selected judge backend."
        ),
        "inspection_safe": False,
        "examples": [
            (
                "umbrela evaluate --backend gpt --model gpt-4o "
                "--qrel dl19-passage --result-file run.trec --output json"
            )
        ],
    },
    "serve": {
        "summary": "Start a FastAPI server for direct umbrela judge requests.",
        "examples": [
            "umbrela serve --backend gpt --model gpt-4o --port 8086",
            (
                "curl -X POST http://127.0.0.1:8086/v1/judge "
                "-H 'content-type: application/json' "
                '-d \'{"query":"q","candidates":["p"]}\''
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8086/v1/judge "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
            (
                "umbrela serve --backend gpt --model gpt-4o "
                "--include-trace --redact-prompts --port 8086"
            ),
        ],
        "routes": ["GET /healthz", "POST /v1/judge"],
        "inspection_safe": True,
    },
    "view": {
        "summary": "Inspect umbrela artifact files with a human-readable preview.",
        "examples": [
            "umbrela view judgments.jsonl",
            "umbrela view judgments.jsonl --records 1 --show-prompts",
        ],
        "supported_types": ["judge-output"],
        "inspection_safe": True,
    },
    "prompt": {
        "summary": "Inspect built-in or custom umbrela prompt templates.",
        "examples": [
            "umbrela prompt list",
            "umbrela prompt show --prompt-type bing --few-shot-count 0",
            (
                "umbrela prompt render --prompt-type basic --input-json "
                '\'{"query":"q","candidates":["p"]}\''
            ),
            (
                "umbrela prompt render --prompt-type basic --few-shot-count 2 "
                "--qrel dl19-passage --input-json "
                '\'{"query":"q","candidates":["p"]}\''
            ),
            "umbrela prompt show --prompt-file custom.yaml --output json",
        ],
        "inspection_safe": True,
        "subcommands": ["list", "show", "render"],
    },
    "describe": {
        "summary": "Inspect structured metadata for a public umbrela command.",
        "inspection_safe": True,
    },
    "schema": {
        "summary": (
            "Print JSON schemas for supported umbrela inputs, outputs, and envelopes."
        ),
        "inspection_safe": True,
    },
    "doctor": {
        "summary": (
            "Report environment, dependency, and backend readiness for "
            "the packaged umbrela CLI."
        ),
        "inspection_safe": True,
    },
    "validate": {
        "summary": (
            "Validate direct JSON input, batch JSONL input, or evaluation "
            "prerequisites without running models."
        ),
        "targets": ["judge", "evaluate"],
        "inspection_safe": True,
    },
}


SCHEMAS: dict[str, dict[str, Any]] = {
    "judge-direct-input": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["text"],
                        "properties": {
                            "qid": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                ]
            },
            "candidates": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "required": ["text"],
                            "properties": {
                                "text": {"type": "string"},
                                "docid": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "required": ["doc"],
                            "properties": {
                                "docid": {"type": "string"},
                                "doc": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "properties": {
                                                "segment": {"type": "string"},
                                                "contents": {"type": "string"},
                                            },
                                        },
                                    ]
                                },
                            },
                        },
                    ]
                },
            },
        },
    },
    "judge-batch-input-record": {
        "type": "object",
        "required": ["query", "candidates"],
        "properties": {
            "query": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "qid": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "candidates": {"type": "array"},
        },
    },
    "judge-output": {
        "type": "object",
        "required": ["query", "passage", "judgment"],
        "properties": {
            "query": {"type": "string"},
            "passage": {"type": "string"},
            "judgment": {"type": "integer"},
            "reasoning": {"type": "string"},
            "prediction": {"type": "string"},
            "result_status": {"type": "integer"},
            "prompt": {"type": "string"},
            "model": {"type": "string"},
        },
    },
    "evaluate-output": {
        "type": "object",
        "required": ["artifacts", "metrics"],
    },
    "view-summary": {
        "type": "object",
        "required": ["path", "artifact_type", "summary", "sampled_records"],
    },
    "prompt-catalog": {
        "type": "array",
        "items": {
            "type": "object",
            "required": [
                "selector",
                "few_shot_mode",
                "source_path",
                "method",
                "placeholders",
            ],
        },
    },
    "prompt-template": {
        "type": "object",
        "required": ["selector", "template"],
    },
    "rendered-prompt": {
        "type": "object",
        "required": ["selector", "messages", "inputs"],
    },
    "doctor-output": {
        "type": "object",
        "required": [
            "python_version",
            "python_ok",
            "env_file_present",
            "backend_readiness",
            "command_readiness",
            "overall_status",
        ],
    },
    "cli-envelope": {
        "type": "object",
        "required": [
            "schema_version",
            "repo",
            "command",
            "mode",
            "status",
            "exit_code",
            "inputs",
            "resolved",
            "artifacts",
            "validation",
            "metrics",
            "warnings",
            "errors",
        ],
    },
}


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
