from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from umbrela.utils import qrel_utils


COMMAND_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "judge": {
        "summary": (
            "Run a single Umbrela backend over direct JSON input "
            "or batch JSONL requests."
        ),
        "execution_mode_default": "sync",
        "backends": ["gpt", "gemini", "hf", "os", "ensemble"],
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
        ],
        "direct_input": {
            "ids_optional": True,
            "shape": {
                "query": "string | {text: string, qid?: string}",
                "candidates": [
                    "string | {text: string, docid?: string} | "
                    "{doc: {segment: string}, docid?: string}"
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
        "examples": [
            (
                "umbrela evaluate --backend gpt --model gpt-4o "
                "--qrel dl19-passage --result-file run.trec --output json"
            )
        ],
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
                                    "type": "object",
                                    "required": ["segment"],
                                    "properties": {"segment": {"type": "string"}},
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
        "required": [
            "model",
            "query",
            "passage",
            "prompt",
            "prediction",
            "judgment",
            "result_status",
        ],
    },
    "evaluate-output": {
        "type": "object",
        "required": ["artifacts", "metrics"],
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
    return {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "env_file_present": env_path.exists(),
        "provider_keys": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "azure": bool(
                os.getenv("AZURE_OPENAI_API_BASE")
                and os.getenv("AZURE_OPENAI_API_VERSION")
            ),
            "gemini": bool(os.getenv("GCLOUD_PROJECT") and os.getenv("GCLOUD_REGION")),
            "huggingface": bool(os.getenv("HF_TOKEN")),
        },
        "pyserini_available": pyserini_available,
        "java_configured": bool(java_home),
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
