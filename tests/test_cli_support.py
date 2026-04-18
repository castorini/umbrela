from __future__ import annotations

import argparse
from typing import Any

import pytest

from umbrela.api.runtime import execute_direct_judge
from umbrela.cli.normalize import normalize_direct_judge_input
from umbrela.cli.responses import CommandResponse

pytestmark = pytest.mark.core


def test_normalize_direct_judge_input_accepts_minimal_payload() -> None:
    normalized = normalize_direct_judge_input(
        {
            "query": "how long is life cycle of flea",
            "candidates": [
                (
                    "The life cycle of a flea can last anywhere "
                    "from 20 days to an entire year."
                )
            ],
        }
    )

    assert normalized["query"] == {
        "qid": "q0",
        "text": "how long is life cycle of flea",
    }
    assert normalized["candidates"] == [
        {
            "docid": "d0",
            "doc": {
                "segment": (
                    "The life cycle of a flea can last anywhere "
                    "from 20 days to an entire year."
                )
            },
        }
    ]


def test_normalize_direct_judge_input_accepts_rich_payload() -> None:
    normalized = normalize_direct_judge_input(
        {
            "query": {"qid": "q7", "text": "anthropological definition of environment"},
            "candidates": [
                {
                    "text": (
                        "Environmental anthropology studies "
                        "human-environment relations."
                    ),
                    "docid": "d9",
                }
            ],
        }
    )

    assert normalized["query"] == {
        "qid": "q7",
        "text": "anthropological definition of environment",
    }
    assert normalized["candidates"][0]["docid"] == "d9"
    assert normalized["candidates"][0]["doc"]["segment"].startswith(
        "Environmental anthropology"
    )


def test_command_response_uses_shared_cli_envelope() -> None:
    response = CommandResponse(command="judge")

    assert response.to_envelope()["schema_version"] == "castorini.cli.v1"
    assert response.to_envelope()["repo"] == "umbrela"


def test_execute_direct_judge_matches_normalized_request_shape() -> None:
    payload = {
        "schema_version": "castorini.cli.v1",
        "repo": "rank_llm",
        "command": "rerank",
        "artifacts": [
            {
                "name": "rerank-results",
                "kind": "data",
                "value": [
                    {
                        "query": {"text": "q", "qid": ""},
                        "candidates": [
                            {
                                "docid": "d0",
                                "score": 1.0,
                                "doc": {"contents": "p"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    expected = normalize_direct_judge_input(payload)
    seen: dict[str, object] = {}

    def fake_judge_runner(
        request_dict: dict[str, Any], args: argparse.Namespace
    ) -> list[dict[str, Any]]:
        seen["request_dict"] = request_dict
        seen["args_model"] = args.model
        return [
            {
                "model": str(args.model),
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    response = execute_direct_judge(
        payload,
        args=argparse.Namespace(
            backend="gpt",
            model="gpt-4o",
            prompt_type="bing",
            few_shot_count=0,
            execution_mode="sync",
            include_reasoning=False,
            include_trace=False,
            redact_prompts=False,
        ),
        judge_runner=fake_judge_runner,
    )

    assert seen["request_dict"] == expected
    assert seen["args_model"] == "gpt-4o"
    assert response.validation["valid"] is True
    assert response.artifacts[0]["data"][0] == {
        "query": "q",
        "passage": "p",
        "judgment": 3,
    }
