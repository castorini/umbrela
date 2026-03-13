from __future__ import annotations

from pathlib import Path

import pytest

from umbrela.utils import common_utils, qrel_utils


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"overall_score": 3}', (3, 1)),
        ("I would give it a score of 2.", (2, 1)),
        ("final score is 1", (1, 1)),
        ("no numeric label present", (0, 0)),
    ],
)
def test_parse_fewshot_response_handles_supported_and_invalid_formats(
    response: str, expected: tuple[int, int]
) -> None:
    assert (
        common_utils.parse_fewshot_response(
            response,
            passage="passage text",
            query="query text",
        )
        == expected
    )


def test_prepare_judgments_logs_invalid_response_with_query_then_passage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    judgments = common_utils.prepare_judgments(
        outputs=["not a valid label"],
        query_passage=[("query text", "passage text")],
        prompts=["prompt text"],
        model_name="fixture-model",
    )

    captured = capsys.readouterr()
    assert "Invalid response to `query text` & `passage text`" in captured.out
    assert judgments[0]["judgment"] == 0
    assert judgments[0]["result_status"] == 0


def test_write_modified_qrel_round_trips_through_qrel_reader(
    tmp_path: Path,
) -> None:
    qrel_path = tmp_path / "sample.qrels"
    modified_qrels: qrel_utils.QrelsData = {
        1: {"d1": "2", "d2": "0"},
        "2": {3: "1"},
    }

    common_utils.write_modified_qrel(
        modified_qrels,
        str(qrel_path),
    )

    assert qrel_utils.get_qrels(str(qrel_path)) == {
        1: {"d1": "2", "d2": "0"},
        2: {3: "1"},
    }
