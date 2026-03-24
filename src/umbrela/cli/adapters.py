from __future__ import annotations

from pathlib import Path
from typing import Any


def make_data_artifact(name: str, data: Any) -> dict[str, Any]:
    return {"name": name, "kind": "data", "data": data}


def make_file_artifact(name: str, path: str) -> dict[str, Any]:
    return {"name": name, "kind": "file", "path": str(Path(path))}


def serialize_direct_judgment(
    judgment: dict[str, Any],
    *,
    include_reasoning: bool,
    include_trace: bool,
    redact_prompts: bool,
) -> dict[str, Any]:
    serialized = {
        "query": judgment["query"],
        "passage": judgment["passage"],
        "judgment": judgment["judgment"],
    }
    if include_reasoning and judgment.get("reasoning"):
        serialized["reasoning"] = judgment["reasoning"]
    if include_trace:
        if "prediction" in judgment:
            serialized["prediction"] = judgment["prediction"]
        if "result_status" in judgment:
            serialized["result_status"] = judgment["result_status"]
        if "prompt" in judgment:
            serialized["prompt"] = (
                "[redacted]" if redact_prompts else judgment["prompt"]
            )
    return serialized
