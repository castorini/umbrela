from __future__ import annotations

from pathlib import Path
from typing import Any


def make_data_artifact(name: str, data: Any) -> dict[str, Any]:
    return {"name": name, "kind": "data", "data": data}


def make_file_artifact(name: str, path: str, *, kind: str = "file") -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "path": str(Path(path)),
    }
