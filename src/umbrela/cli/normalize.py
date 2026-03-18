from __future__ import annotations

from typing import Any, cast


def _normalize_query(query: Any) -> dict[str, str]:
    if isinstance(query, str):
        return {"qid": "q0", "text": query}
    if isinstance(query, dict) and isinstance(query.get("text"), str):
        normalized = {"text": cast(str, query["text"])}
        normalized["qid"] = str(query.get("qid", "q0"))
        return normalized
    raise ValueError(
        "Direct judge input requires `query` as a string or object with `text`."
    )


def _normalize_candidate(candidate: Any, index: int) -> dict[str, Any]:
    if isinstance(candidate, str):
        return {"docid": f"d{index}", "doc": {"segment": candidate}}
    if isinstance(candidate, dict):
        if isinstance(candidate.get("doc"), dict):
            segment = candidate["doc"].get("segment") or candidate["doc"].get(
                "contents"
            )
            if isinstance(segment, str):
                return {
                    "docid": str(candidate.get("docid", f"d{index}")),
                    "doc": {"segment": segment},
                }
        if isinstance(candidate.get("text"), str):
            return {
                "docid": str(candidate.get("docid", f"d{index}")),
                "doc": {"segment": candidate["text"]},
            }
    raise ValueError(
        "Direct judge input candidates must be strings or "
        "objects containing `text`, `doc.segment`, or `doc.contents`."
    )


def normalize_direct_judge_input(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Direct judge input requires a non-empty `candidates` array.")
    return {
        "query": _normalize_query(payload.get("query")),
        "candidates": [
            _normalize_candidate(candidate, index)
            for index, candidate in enumerate(candidates)
        ],
    }
