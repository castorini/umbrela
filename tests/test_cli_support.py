from __future__ import annotations

from umbrela.cli.normalize import normalize_direct_judge_input
from umbrela.cli.responses import CommandResponse


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
