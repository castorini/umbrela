from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import umbrela.cli.introspection as introspection
from umbrela.cli.main import main

pytestmark = pytest.mark.core


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


def test_batch_json_output_suppresses_progress_bar(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
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
        return [
            {
                "model": "gpt-4o",
                "query": "q",
                "passage": "p",
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
            "gpt",
            "--model",
            "gpt-4o",
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_quiet_flag_suppresses_stderr(capsys: Any) -> None:
    exit_code = main(["--quiet", "doctor", "--output", "json"])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_no_color_env_suppresses_ansi_codes(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    path = tmp_path / "judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "model": "gpt-4o",
                "query": "query",
                "passage": "passage",
                "prompt": "prompt",
                "prediction": "##final score: 3",
                "reasoning": None,
                "judgment": 3,
                "result_status": 1,
            }
        ],
    )

    exit_code = main(["view", str(path), "--color", "always"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "\033[" not in stdout


def test_print_completion_outputs_bash_script(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--print-completion", "bash"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "complete" in stdout.lower() or "_umbrela" in stdout


def test_version_flag_prints_version_and_exits(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "umbrela" in stdout


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
    assert output["artifacts"][0]["kind"] == "data"
    assert output["artifacts"][0]["name"] == "judgments"
    assert output["artifacts"][0]["data"][0]["judgment"] == 3
    assert (
        output["artifacts"][0]["data"][0]["query"] == "how long is life cycle of flea"
    )
    assert output["artifacts"][0]["data"][0]["passage"].startswith(
        "The life cycle of a flea can last"
    )
    assert "prediction" not in output["artifacts"][0]["data"][0]
    assert "result_status" not in output["artifacts"][0]["data"][0]
    assert "prompt" not in output["artifacts"][0]["data"][0]
    assert "reasoning" not in output["artifacts"][0]["data"][0]
    assert "normalized_request" not in output["resolved"]
    assert output["resolved"]["backend"] == "gpt"
    assert output["resolved"]["model"] == "gpt-4o"
    assert output["resolved"]["input_mode"] == "direct"


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
    assert "prediction" not in output["artifacts"][0]["data"][0]


def test_direct_judge_can_include_trace(monkeypatch: Any, capsys: Any) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict, args
        return [
            {
                "model": "gpt-4o",
                "query": "q",
                "passage": "p",
                "prompt": "prompt",
                "prediction": "2",
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
            "--include-trace",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    record = output["artifacts"][0]["data"][0]
    assert record["prediction"] == "2"
    assert record["result_status"] == 1
    assert record["prompt"] == "prompt"


def test_direct_judge_can_redact_prompts_when_trace_enabled(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict, args
        return [
            {
                "model": "gpt-4o",
                "query": "q",
                "passage": "p",
                "prompt": "very secret prompt",
                "prediction": "2",
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
            "--include-trace",
            "--redact-prompts",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"][0]["prompt"] == "[redacted]"


def test_direct_judge_accepts_anserini_rest_payload(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        assert args.model == "gpt-4o"
        assert request_dict["query"]["text"] == "what is python"
        assert request_dict["query"]["qid"] == "q0"
        assert request_dict["candidates"][0]["docid"] == "1737459"
        assert (
            request_dict["candidates"][0]["doc"]["segment"]
            == "Python is widely used for web development."
        )
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "3",
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
                    "api": "v1",
                    "index": "msmarco-v1-passage",
                    "query": {"text": "what is python"},
                    "candidates": [
                        {
                            "docid": "1737459",
                            "score": 10.58,
                            "rank": 1,
                            "doc": "Python is widely used for web development.",
                        }
                    ],
                }
            ),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"


def test_direct_judge_text_output_uses_query_candidate_judgment_reasoning(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del args
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
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "query: q\ncandidate: p\njudgment: 2\nreasoning: reasoning content\n"
    )


def test_direct_judge_text_output_marks_parsing_failure(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del args
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "0",
                "judgment": 0,
                "result_status": 0,
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
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "query: q\ncandidate: p\njudgment: 0\nparsing failed\n"
    )


def test_direct_judge_text_output_separates_multiple_records(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict, args
        return [
            {
                "model": "gpt-4o",
                "query": "q1",
                "passage": "p1",
                "prompt": "prompt",
                "prediction": "1",
                "judgment": 1,
                "result_status": 1,
            },
            {
                "model": "gpt-4o",
                "query": "q2",
                "passage": "p2",
                "prompt": "prompt",
                "prediction": "2",
                "judgment": 2,
                "result_status": 1,
            },
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
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "query: q1\ncandidate: p1\njudgment: 1\n-----\n"
        "query: q2\ncandidate: p2\njudgment: 2\n"
    )


def test_direct_judge_accepts_minimal_reasoning_effort(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict
        assert args.reasoning_effort == "minimal"
        return [
            {
                "model": "gpt-5.4",
                "query": "q",
                "passage": "p",
                "prompt": "prompt",
                "prediction": "2",
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
            "gpt-5.4",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--reasoning-effort",
            "minimal",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifacts"][0]["data"][0]["judgment"] == 2


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


def test_cli_and_serve_normalize_direct_payloads_identically(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

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
    seen: dict[str, Any] = {}

    def fake_cli_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        seen["cli_request"] = request_dict
        del args
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    def fake_api_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        seen["api_request"] = request_dict
        del args
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_direct", fake_cli_run_judge_direct)
    monkeypatch.setattr(
        "umbrela.api.runtime.run_judge_direct", fake_api_run_judge_direct
    )

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-json",
            json.dumps(payload),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )
    response = client.post("/v1/judge", json=payload)

    assert response.status_code == 200
    assert seen["cli_request"] == seen["api_request"]


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


def test_batch_judge_can_write_filtered_requests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    input_path = tmp_path / "requests.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    filtered_path = tmp_path / "relevant.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "query": {"qid": "q1", "text": "what is python used for"},
                "candidates": [
                    {
                        "docid": "d1",
                        "doc": {"segment": "Python is used for web development."},
                    },
                    {
                        "docid": "d2",
                        "doc": {"segment": "Python is a kind of snake."},
                    },
                ],
            }
        ],
    )

    def fake_run_judge_batch(
        records: list[dict[str, Any]], args: Any
    ) -> list[dict[str, Any]]:
        assert len(records) == 1
        assert args.min_judgment == 2
        return [
            {
                "model": "gemini-1.5-pro",
                "query": "what is python used for",
                "passage": "Python is used for web development.",
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            },
            {
                "model": "gemini-1.5-pro",
                "query": "what is python used for",
                "passage": "Python is a kind of snake.",
                "prompt": "prompt",
                "prediction": "1",
                "judgment": 1,
                "result_status": 1,
            },
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
            str(judgments_path),
            "--filtered-output-file",
            str(filtered_path),
            "--min-judgment",
            "2",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    filtered_records = read_jsonl(filtered_path)
    assert len(filtered_records) == 1
    assert filtered_records[0]["query"]["qid"] == "q1"
    assert [candidate["docid"] for candidate in filtered_records[0]["candidates"]] == [
        "d1"
    ]


def test_batch_judge_text_output_is_quiet_when_writing_output_file(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
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
        ]
    )

    assert exit_code == 0
    assert read_jsonl(output_path)[0]["judgment"] == 3
    assert capsys.readouterr().out == ""


def test_batch_judge_filtered_output_requires_min_judgment(capsys: Any) -> None:
    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-file",
            "/tmp/does-not-exist.jsonl",
            "--filtered-output-file",
            "/tmp/relevant.jsonl",
            "--output",
            "json",
        ]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "validation_error"
    assert output["errors"][0]["code"] == "missing_min_judgment"


def test_batch_judge_filtered_output_fails_when_threshold_removes_all_candidates(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    input_path = tmp_path / "requests.jsonl"
    filtered_path = tmp_path / "relevant.jsonl"
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
        return [
            {
                "model": "gpt-4o",
                "query": "what is python used for",
                "passage": "Python is used for web development.",
                "prompt": "prompt",
                "prediction": "1",
                "judgment": 1,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.cli.main.run_judge_batch", fake_run_judge_batch)

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-file",
            str(input_path),
            "--filtered-output-file",
            str(filtered_path),
            "--min-judgment",
            "2",
            "--output",
            "json",
        ]
    )

    assert exit_code == 6
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "runtime_error"
    assert output["errors"][0]["code"] == "empty_filtered_request"


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
                "query": "Q" * 100,
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
    assert output["artifacts"][0]["data"]["sampled_records"][0]["query"] == "Q" * 100


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
    assert "umbrela View" in stdout
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


def test_view_compact_judgments_text_defaults_status_and_hides_prediction(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "compact_judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "query",
                "passage": "passage",
                "judgment": 3,
            }
        ],
    )

    exit_code = main(["view", str(path), "--color", "never"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "score=3 status=1" in stdout
    assert "prediction:" not in stdout


def test_view_compact_judgments_json_summary_omits_prediction_and_prompt(
    tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "compact_judgments.jsonl"
    write_jsonl(
        path,
        [
            {
                "query": "query",
                "passage": "passage",
                "judgment": 2,
            }
        ],
    )

    exit_code = main(["view", str(path), "--records", "1", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    sampled = output["artifacts"][0]["data"]["sampled_records"][0]
    assert sampled["result_status"] == 1
    assert "prediction" not in sampled
    assert "prompt" not in sampled


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
    assert (
        "judge, evaluate, serve, view, prompt, describe, schema, doctor, validate"
        in captured.err
    )
    assert "Run `umbrela --help` for full usage." in captured.err


def test_describe_judge_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["describe", "judge", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "describe"
    assert "backends" in output["artifacts"][0]["data"]
    assert output["artifacts"][0]["data"]["backends"] == ["gpt", "gemini", "hf", "os"]


def test_schema_judge_direct_input_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["schema", "judge-direct-input", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "schema"
    assert "query" in output["artifacts"][0]["data"]["properties"]
    assert "overrides" in output["artifacts"][0]["data"]["properties"]


def test_prompt_list_returns_json_catalog(capsys: Any) -> None:
    exit_code = main(["prompt", "list", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["artifacts"][0]["name"] == "prompt-catalog"
    catalog = output["artifacts"][0]["data"]
    assert len(catalog) == 4
    assert catalog[0]["selector"]["prompt_type"] in {"basic", "bing"}


def test_prompt_show_builtin_returns_text_template(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "show",
            "--prompt-type",
            "bing",
            "--few-shot-count",
            "0",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "umbrela Prompt Template" in stdout
    assert "prompt_type: bing" in stdout
    assert "few_shot_count: 0" in stdout
    assert "[user]" in stdout
    assert "Query: {query}" in stdout


def test_prompt_show_custom_returns_json_template(tmp_path: Path, capsys: Any) -> None:
    prompt_path = tmp_path / "custom_prompt.yaml"
    prompt_path.write_text(
        'method: "custom"\n'
        'system_message: "custom system"\n'
        'prefix_user: "Examples: {examples}\\nQuery: {query}\\nPassage: {passage}"\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "prompt",
            "show",
            "--prompt-file",
            str(prompt_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    template = output["artifacts"][0]["data"]["template"]
    assert template["method"] == "custom"
    assert template["system_message"] == "custom system"
    assert template["placeholders"] == ["examples", "query", "passage"]


def test_prompt_show_requires_template_selector(capsys: Any) -> None:
    exit_code = main(["prompt", "show", "--output", "json"])

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "unknown"
    assert output["errors"][0]["code"] == "invalid_arguments"


def test_prompt_render_builtin_returns_text_prompt(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p0", "p1"]}),
            "--candidate-index",
            "1",
            "--part",
            "user",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "umbrela Rendered Prompt" in stdout
    assert "prompt_type: basic" in stdout
    assert "candidate_index: 1" in stdout
    assert "passage: p1" in stdout
    assert "[system]" not in stdout
    assert "[user]" in stdout
    assert "Query: q" in stdout
    assert "Passage: p1" in stdout


def test_prompt_render_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "bing",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    rendered = output["artifacts"][0]["data"]
    assert rendered["messages"]["system"] == ""
    assert "Query: q" in rendered["messages"]["user"]
    assert "Passage: p" in rendered["messages"]["user"]


def test_prompt_render_few_shot_uses_examples_text(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--few-shot-count",
            "2",
            "--examples-text",
            "demo examples",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "few_shot_count: 2" in stdout
    assert "demo examples" in stdout


def test_prompt_render_few_shot_uses_examples_file(tmp_path: Path, capsys: Any) -> None:
    examples_path = tmp_path / "examples.txt"
    examples_path.write_text("examples from file", encoding="utf-8")

    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--few-shot-count",
            "2",
            "--examples-file",
            str(examples_path),
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    rendered = output["artifacts"][0]["data"]
    assert rendered["inputs"]["examples"] == "examples from file"


def test_prompt_render_few_shot_generates_examples_from_qrel(
    monkeypatch: Any, capsys: Any
) -> None:
    def fake_generate_examples_prompt(qrel: str, few_shot_count: int) -> str:
        assert qrel == "dl19-passage"
        assert few_shot_count == 2
        return "generated examples"

    monkeypatch.setattr(
        "umbrela.cli.main.qrel_utils.generate_examples_prompt",
        fake_generate_examples_prompt,
    )

    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--few-shot-count",
            "2",
            "--qrel",
            "dl19-passage",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    rendered = output["artifacts"][0]["data"]
    assert rendered["selector"]["qrel"] == "dl19-passage"
    assert rendered["inputs"]["examples"] == "generated examples"


def test_prompt_render_few_shot_requires_qrel_or_examples(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--few-shot-count",
            "2",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["errors"][0]["code"] == "missing_prompt_examples"


def test_prompt_render_reports_example_generation_failure(
    monkeypatch: Any, capsys: Any
) -> None:
    def fail_generate_examples_prompt(qrel: str, few_shot_count: int) -> str:
        del qrel, few_shot_count
        raise RuntimeError("pyserini missing")

    monkeypatch.setattr(
        "umbrela.cli.main.qrel_utils.generate_examples_prompt",
        fail_generate_examples_prompt,
    )

    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--few-shot-count",
            "2",
            "--qrel",
            "dl19-passage",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--output",
            "json",
        ]
    )

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["errors"][0]["code"] == "prompt_example_generation_failed"


def test_prompt_render_rejects_invalid_candidate_index(capsys: Any) -> None:
    exit_code = main(
        [
            "prompt",
            "render",
            "--prompt-type",
            "basic",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--candidate-index",
            "2",
            "--output",
            "json",
        ]
    )

    assert exit_code == 5
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "prompt"
    assert output["errors"][0]["code"] == "invalid_candidate_index"


def test_doctor_returns_json_envelope(capsys: Any) -> None:
    exit_code = main(["doctor", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "doctor"
    assert "python_version" in output["metrics"]
    assert "openrouter" in output["metrics"]["provider_keys"]
    assert "backend_readiness" in output["metrics"]
    assert "serve" in output["metrics"]["command_readiness"]


def test_doctor_reports_missing_dependencies(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GCLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GCLOUD_REGION", "europe-west4")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    available_modules = {"torch"}

    def fake_find_spec(name: str) -> object | None:
        return object() if name in available_modules else None

    monkeypatch.setattr(introspection.importlib.util, "find_spec", fake_find_spec)

    exit_code = main(["doctor", "--output", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    backend_readiness = output["metrics"]["backend_readiness"]
    assert backend_readiness["gpt"]["status"] == "missing_dependency"
    assert backend_readiness["gpt"]["missing_dependencies"] == ["openai"]
    assert backend_readiness["gemini"]["status"] == "missing_dependency"
    assert backend_readiness["gemini"]["missing_dependencies"] == ["vertexai"]
    assert backend_readiness["hf"]["status"] == "missing_dependency"
    assert backend_readiness["hf"]["missing_dependencies"] == ["transformers"]
    assert backend_readiness["os"]["status"] == "missing_dependency"
    assert backend_readiness["os"]["missing_dependencies"] == [
        "transformers",
        "fastchat",
    ]


def test_serve_command_starts_uvicorn(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    seen: dict[str, Any] = {}

    def fake_run(app: Any, host: str, port: int) -> None:
        seen["app"] = app
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)

    exit_code = main(
        ["serve", "--backend", "gpt", "--model", "gpt-4o", "--port", "8084"]
    )

    assert exit_code == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 8084


def test_serve_app_health_and_judge(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        return [
            {
                "model": args.model,
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.api.runtime.run_judge_direct", fake_run_judge_direct)

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )

    health_response = client.get("/healthz")
    judge_response = client.post(
        "/v1/judge",
        json={
            "api": "v1",
            "index": "msmarco-v1-passage",
            "query": {"text": "q"},
            "candidates": [
                {
                    "docid": "d0",
                    "score": 1.0,
                    "rank": 1,
                    "doc": "p",
                }
            ],
        },
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert judge_response.status_code == 200
    envelope = judge_response.json()
    assert envelope["artifacts"][0]["name"] == "judgments"
    assert "prediction" not in envelope["artifacts"][0]["data"][0]
    assert "prompt" not in envelope["artifacts"][0]["data"][0]
    assert "result_status" not in envelope["artifacts"][0]["data"][0]
    assert "normalized_request" not in envelope["resolved"]


def test_serve_app_can_include_trace_and_redact_prompts(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict, args
        return [
            {
                "model": "gpt-4o",
                "query": "q",
                "passage": "p",
                "prompt": "very secret prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.api.runtime.run_judge_direct", fake_run_judge_direct)

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
                include_trace=True,
                redact_prompts=True,
            )
        )
    )

    response = client.post("/v1/judge", json={"query": "q", "candidates": ["p"]})

    assert response.status_code == 200
    record = response.json()["artifacts"][0]["data"][0]
    assert record["prediction"] == "3"
    assert record["result_status"] == 1
    assert record["prompt"] == "[redacted]"


def test_serve_app_rejects_invalid_payload() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )

    response = client.post("/v1/judge", json={"query": 1, "candidates": []})

    assert response.status_code == 400
    assert response.json()["status"] == "validation_error"


def test_serve_app_applies_request_overrides(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    captured: dict[str, Any] = {}

    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict
        captured["backend"] = args.backend
        captured["model"] = args.model
        captured["reasoning_effort"] = args.reasoning_effort
        return [
            {
                "model": args.model,
                "query": "q",
                "passage": "p",
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.api.runtime.run_judge_direct", fake_run_judge_direct)

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )

    response = client.post(
        "/v1/judge",
        json={
            "query": "q",
            "candidates": ["p"],
            "overrides": {
                "backend": "gpt",
                "model": "gpt-4.1-mini",
                "reasoning_effort": "low",
            },
        },
    )

    assert response.status_code == 200
    assert captured == {
        "backend": "gpt",
        "model": "gpt-4.1-mini",
        "reasoning_effort": "low",
    }
    assert response.json()["resolved"]["model"] == "gpt-4.1-mini"


def test_serve_app_rejects_invalid_override_combinations() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )

    response = client.post(
        "/v1/judge",
        json={
            "query": "q",
            "candidates": ["p"],
            "overrides": {
                "backend": "gemini",
                "use_openrouter": True,
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["status"] == "validation_error"


def test_serve_app_accepts_rank_llm_envelope_payload(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from umbrela.api.app import create_app
    from umbrela.api.runtime import ServerConfig

    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del request_dict, args
        return [
            {
                "model": "gpt-4o",
                "query": "q",
                "passage": "p",
                "prompt": "prompt",
                "prediction": "3",
                "judgment": 3,
                "result_status": 1,
            }
        ]

    monkeypatch.setattr("umbrela.api.runtime.run_judge_direct", fake_run_judge_direct)

    client = TestClient(
        create_app(
            ServerConfig(
                host="127.0.0.1",
                port=8084,
                backend="gpt",
                model="gpt-4o",
            )
        )
    )

    response = client.post(
        "/v1/judge",
        json={
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
        },
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["name"] == "judgments"


def test_top_level_help_includes_command_summaries(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert "umbrela packaged CLI" in stdout
    assert "run a single judge backend" in stdout.lower()
    assert "inspect an existing umbrela artifact" in stdout.lower()


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
    assert output["mode"] == "dry-run"


def test_judge_validate_only_returns_validated_request_without_running_backend(
    monkeypatch: Any, capsys: Any
) -> None:
    def fail_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        raise AssertionError("judge backend should not run in validate-only mode")

    monkeypatch.setattr("umbrela.cli.main.run_judge_direct", fail_run_judge_direct)

    exit_code = main(
        [
            "judge",
            "--backend",
            "gpt",
            "--model",
            "gpt-4o",
            "--input-json",
            json.dumps({"query": "q", "candidates": ["p"]}),
            "--validate-only",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "validate"
    assert output["artifacts"][0]["name"] == "validated-request"


def test_judge_manifest_path_writes_envelope(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    manifest_path = tmp_path / "judge-manifest.json"

    def fake_run_judge_direct(
        request_dict: dict[str, Any], args: Any
    ) -> list[dict[str, Any]]:
        del args
        return [
            {
                "model": "gpt-4o",
                "query": request_dict["query"]["text"],
                "passage": request_dict["candidates"][0]["doc"]["segment"],
                "prompt": "prompt",
                "prediction": "2",
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
            "--manifest-path",
            str(manifest_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["command"] == "judge"
    assert manifest == output


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


def test_config_file_sets_default_output_format(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    config_dir = tmp_path / "config" / "umbrela"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text('output = "json"\n', encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    exit_code = main(["doctor"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    output = json.loads(stdout)
    assert output["command"] == "doctor"
    assert output["metrics"]["config_file"] == str(config_file)


def test_pipe_judge_jsonl_output_is_valid_jsonl(monkeypatch: Any, capsys: Any) -> None:
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
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    envelope = json.loads(stdout)
    assert envelope["schema_version"] == "castorini.cli.v1"
    assert all(isinstance(record, dict) for record in envelope["artifacts"][0]["data"])
