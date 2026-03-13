from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from umbrela.cli.main import main


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_direct_judge_via_input_json(monkeypatch: Any, capsys: Any) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        assert request_dict["query"]["qid"] == "q0"
        assert request_dict["candidates"][0]["docid"] == "d0"
        assert args.backend == "gpt"
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "##final score: 3",
                "reasoning": "reasoning content",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_direct", fake_run_judge_direct)

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-json",
            json.dumps(
                {
                    "query": "how long is life cycle of flea",
                    "candidates": [
                        (
                            "The life cycle of a flea can last anywhere "
                            "from 20 days to an entire year."
                        )
                    ],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "judge"
    assert output["schema_version"] == "castorini.cli.v1"
    assert output["artifacts"][0]["data"][0]["judgment"] == 3
    assert "reasoning" not in output["artifacts"][0]["data"][0]


def test_direct_judge_can_include_reasoning(monkeypatch: Any, capsys: Any) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "2",
                "reasoning": "reasoning content",
                "judgment": 2,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_direct", fake_run_judge_direct)

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--include-reasoning",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"][0]["reasoning"] == "reasoning content"


def test_direct_judge_via_stdin(monkeypatch: Any, capsys: Any) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "1",
                "judgment": 1,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_direct", fake_run_judge_direct)
    monkeypatch.setattr(
        "sys.stdin.read",
        lambda: json.dumps({"query": "q", "candidates": ["p"]}),
    )

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--stdin",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"][0]["judgment"] == 1


def test_batch_judge_writes_jsonl_output(tmp_path: Path, monkeypatch: Any) -> None:
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "judgments.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": {"qid": "q1", "text": "what is python used for"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {"segment": "Python is used for web development."},
                    }
                ],
            }
        ],
    )

    def fake_run_judge_batch(
        records: list[dict[str, Any]], args: Any
    ) -> list[dict[str, Any]]:
        assert len(records) == 1
        assert args.backend == "gemini"
        return [
            {
                "model": "gemini-1.5-pro",
                "query": "what is python used for",
                "passage": "Python is used for web development.",
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_batch", fake_run_judge_batch)

    exit_code = main(
        [
            "judge",
            "--backend",
            "gemini",
            "--model",
            "gemini-1.5-pro",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    records = read_jsonl(output_path)
    assert records[0]["judgment"] == 3


def test_batch_judge_missing_input_returns_json_error(capsys: Any) -> None:
    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-file",
            "/tmp/does-not-exist.jsonl",
            "--output",
            "json",
        ]
    )

    assert exit_code == 4
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "validation_error"
    assert output["errors"][0]["code"] == "missing_input"


def test_missing_command_returns_descriptive_text_error(capsys: Any) -> None:
    exit_code = main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No command provided." in captured.err
    assert "judge, evaluate, describe, schema, doctor, validate" in captured.err
    assert "Run `umbrela --help` for full usage." in captured.err


def test_describe_judge_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["describe", "judge", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "describe"
    assert "backends" in output["artifacts"][0]["data"]


def test_schema_judge_direct_input_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["schema", "judge-direct-input", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "schema"
    assert "query" in output["artifacts"][0]["data"]["properties"]


def test_doctor_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["doctor", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "doctor"
    assert "python_version" in output["metrics"]


def test_validate_judge_direct_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(
        [
            "validate",
            "judge",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p1", "p2"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "validate"
    assert output["validation"]["valid"] is True


def test_evaluate_dry_run_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(
        [
            "evaluate",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--qrel",
            "dl19-passage",
            "--result-file",
            __file__,
            "--dry-run",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "evaluate"
    assert output["mode"] == "dry_run"


def test_async_mode_rejected_for_non_gpt(capsys: Any) -> None:
    exit_code = main(
        [
            "judge",
            "--backend",
            "gemini",
            "--model",
            "gemini-1.5-pro",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--execution-mode",
            "async",
            "--output",
            "json",
        ]
    )

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["errors"][0]["code"] == "unsupported_execution_mode"
