from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}

JUDGE_REQUIRED_KEYS = {
    "model",
    "query",
    "passage",
    "prediction",
    "judgment",
    "result_status",
}


class ViewError(ValueError):
    """Raised when a file cannot be viewed as a supported Umbrela artifact."""


def _color_enabled(color: str) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if color == "always":
        return True
    if color == "never":
        return False
    return sys.stdout.isatty()


def _style(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI_CODES[color]}{text}{ANSI_CODES['reset']}"


def _truncate(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "\u2026"


def load_records(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ViewError(f"path does not exist: {path}") from exc

    if not raw_text.strip():
        raise ViewError(f"file is empty: {path}")

    try:
        if file_path.suffix == ".json":
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return payload
            raise ViewError(f"unsupported JSON payload in {path}")
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ViewError(f"file is not valid JSON/JSONL: {path}") from exc

    if not records:
        raise ViewError(f"file is empty: {path}")
    return records


def detect_artifact_type(
    records: list[dict[str, Any]], requested_type: str | None
) -> str:
    if requested_type is not None:
        if requested_type != "judge-output":
            raise ViewError(
                "unsupported --type for umbrela view; expected judge-output"
            )
        return requested_type

    first_record = records[0]
    if JUDGE_REQUIRED_KEYS.issubset(first_record.keys()):
        return "judge-output"
    raise ViewError("could not detect Umbrela artifact type; use --type judge-output")


def summarize_judgments(records: list[dict[str, Any]]) -> dict[str, Any]:
    histogram = {str(score): 0 for score in range(4)}
    invalid_count = 0
    prompt_count = 0
    for record in records:
        judgment = record.get("judgment")
        if judgment in (0, 1, 2, 3):
            histogram[str(judgment)] += 1
        if int(record.get("result_status", 0)) == 0:
            invalid_count += 1
        if record.get("prompt"):
            prompt_count += 1
    return {
        "record_count": len(records),
        "score_histogram": histogram,
        "invalid_count": invalid_count,
        "prompt_count": prompt_count,
    }


def build_view_summary(
    path: str,
    records: list[dict[str, Any]],
    artifact_type: str,
    *,
    record_limit: int,
    show_prompts: bool,
) -> dict[str, Any]:
    limit = max(record_limit, 0)
    sampled_records: list[dict[str, Any]] = []
    for record in records[:limit]:
        item = {
            "judgment": record["judgment"],
            "result_status": record["result_status"],
            "query": _truncate(str(record["query"]), 140),
            "passage": _truncate(str(record["passage"]), 160),
            "prediction": _truncate(str(record["prediction"]), 120),
        }
        if show_prompts and record.get("prompt"):
            item["prompt"] = _truncate(str(record["prompt"]), 240)
        sampled_records.append(item)

    return {
        "path": str(Path(path)),
        "artifact_type": artifact_type,
        "summary": summarize_judgments(records),
        "sampled_records": sampled_records,
        "requested_records": limit,
        "show_prompts": show_prompts,
    }


def render_view_summary(view: dict[str, Any], *, color: str) -> str:
    enabled = _color_enabled(color)
    lines = [
        _style("Umbrela View", "bold", enabled),
        f"path: {view['path']}",
        f"type: {view['artifact_type']}",
        f"records: {view['summary']['record_count']}",
    ]
    histogram = view["summary"]["score_histogram"]
    histogram_text = ", ".join(f"{score}={count}" for score, count in histogram.items())
    lines.append(f"scores: {histogram_text}")
    lines.append(f"invalid: {view['summary']['invalid_count']}")

    for index, record in enumerate(view["sampled_records"], start=1):
        status_value = int(record["result_status"])
        score_text = _style(str(record["judgment"]), "green", enabled)
        status_text = _style(
            str(status_value),
            "yellow" if status_value else "red",
            enabled,
        )
        lines.append("")
        lines.append(f"[{index}] score={score_text} status={status_text}")
        lines.append(f"query: {record['query']}")
        lines.append(f"passage: {record['passage']}")
        lines.append(f"prediction: {record['prediction']}")
        if "prompt" in record:
            lines.append(f"prompt: {record['prompt']}")
    return "\n".join(lines)
