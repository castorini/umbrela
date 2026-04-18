from __future__ import annotations

from typing import Any

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
            "umbrela serve --backend gpt --model gpt-4o --port 8084",
            (
                "curl -X POST http://127.0.0.1:8084/v1/judge "
                "-H 'content-type: application/json' "
                '-d \'{"query":"q","candidates":["p"]}\''
            ),
            (
                "curl -X POST http://127.0.0.1:8084/v1/judge "
                "-H 'content-type: application/json' "
                '-d \'{"query":"q","candidates":["p"],'
                '"overrides":{"backend":"gpt","model":"gpt-4.1-mini",'
                '"reasoning_effort":"low"}}\''
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8084/v1/judge "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
            (
                'curl -s "http://127.0.0.1:8081/v1/msmarco-v1-passage/search?query=q" '
                "| curl -s -X POST http://127.0.0.1:8082/v1/rerank "
                '-H "content-type: application/json" --data-binary @- '
                "| curl -s -X POST http://127.0.0.1:8084/v1/judge "
                '-H "content-type: application/json" --data-binary @- | jq'
            ),
            (
                "umbrela serve --backend gpt --model gpt-4o "
                "--include-trace --redact-prompts --port 8084"
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
            "overrides": {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["gpt", "gemini", "hf", "os"],
                    },
                    "model": {"type": "string"},
                    "prompt_file": {"type": "string"},
                    "prompt_type": {"type": "string"},
                    "few_shot_count": {"type": "integer"},
                    "execution_mode": {"type": "string", "enum": ["sync", "async"]},
                    "max_concurrency": {"type": "integer"},
                    "use_azure_openai": {"type": "boolean"},
                    "use_openrouter": {"type": "boolean"},
                    "reasoning_effort": {
                        "type": "string",
                        "enum": ["none", "minimal", "low", "medium", "high", "xhigh"],
                    },
                    "device": {"type": "string"},
                    "include_reasoning": {"type": "boolean"},
                    "include_trace": {"type": "boolean"},
                    "redact_prompts": {"type": "boolean"},
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
