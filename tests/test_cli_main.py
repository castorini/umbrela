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


def test_view_judgments_returns_json_summary(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "model": "gpt-4o",
                "query": "Q" * 180,
                "passage": "P" * 120,
                "prompt": "prompt text " * 30,
                "prediction": "##final score: 2",
                "reasoning": None,
                "judgment": 2,
                "result_status": 1,
            },
            {
                "model": "gpt-4o",
                "query": "second query",
                "passage": "second passage",
                "prompt": "hidden prompt",
                "prediction": "bad output",
                "reasoning": None,
                "judgment": 0,
                "result_status": 0,
            },
        ],
    )

    exit_code = main(["view", str(path), "--records", "1", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["artifacts"][0]["data"]["artifact_type"] == "judge-output"
    assert output["artifacts"][0]["data"]["summary"]["record_count"] == 2
    assert output["artifacts"][0]["data"]["summary"]["score_histogram"] == {
        "0": 1,
        "1": 0,
        "2": 1,
        "3": 0,
    }
    assert len(output["artifacts"][0]["data"]["sampled_records"]) == 1
    assert "prompt" not in output["artifacts"][0]["data"]["sampled_records"][0]
    assert output["artifacts"][0]["data"]["sampled_records"][0]["query"].endswith("...")


def test_view_judgments_text_hides_prompts_by_default(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "model": "gpt-4o",
                "query": "query",
                "passage": "passage",
                "prompt": "very secret prompt",
                "prediction": "##final score: 3",
                "reasoning": None,
                "judgment": 3,
                "result_status": 1,
            }
        ],
    )

    exit_code = main(["view", str(path), "--color", "never"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Umbrela View" in stdout
    assert "scores: 0=0, 1=0, 2=0, 3=1" in stdout
    assert "prompt:" not in stdout


def test_view_judgments_text_can_show_prompts(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "model": "gpt-4o",
                "query": "query",
                "passage": "passage",
                "prompt": "very secret prompt",
                "prediction": "##final score: 3",
                "reasoning": None,
                "judgment": 3,
                "result_status": 1,
            }
        ],
    )

    exit_code = main(["view", str(path), "--show-prompts", "--color", "never"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "prompt: very secret prompt" in stdout


def test_view_empty_file_returns_json_error(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    exit_code = main(["view", str(path), "--output", "json"])

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["errors"][0]["code"] == "invalid_view_input"


def test_view_malformed_file_returns_json_error(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    exit_code = main(["view", str(path), "--output", "json"])

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "view"
    assert output["errors"][0]["code"] == "invalid_view_input"


def test_missing_command_returns_descriptive_text_error(capsys: Any) -> None:
    exit_code = main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "No command provided." in captured.err
    assert "judge, evaluate, view, describe, schema, doctor, validate" in captured.err
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
    assert "openrouter" in output["metrics"]["provider_keys"]


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
