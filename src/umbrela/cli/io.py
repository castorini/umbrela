from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
