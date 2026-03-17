from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from umbrela.cli import operations
from umbrela.cli.main import main
from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils, qrel_utils

pytestmark = pytest.mark.core


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evaluate"


class ScriptedJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        prompt_file: Path,
        scripted_outputs: list[str],
    ) -> None:
        super().__init__(
            qrel=qrel,
            model_name="fixture/model",
            prompt_file=str(prompt_file),
            prompt_type=None,
            few_shot_count=0,
        )
        self.scripted_outputs = scripted_outputs

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        raise AssertionError("ScriptedJudge.predict_with_llm should not be called")

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        del max_new_tokens
        query_passage, prompts = self.prepare_request_inputs(request_dict, preprocess)
        judgments: list[common_utils.Judgment] = []
        for (query, passage), prompt, output in zip(
            query_passage,
            prompts,
            self.scripted_outputs,
            strict=True,
        ):
            label, status = common_utils.parse_fewshot_response(output, passage, query)
            judgments.append(
                {
                    "model": self.model_name,
                    "query": query,
                    "passage": passage,
                    "prompt": prompt,
                    "prediction": output,
                    "reasoning": None,
                    "judgment": label,
                    "result_status": status,
                }
            )
        return judgments


class NoJudgeReuse(LLMJudge):
    def __init__(self, qrel: str, prompt_file: Path) -> None:
        super().__init__(
            qrel=qrel,
            model_name="fixture/model",
            prompt_file=str(prompt_file),
            prompt_type=None,
            few_shot_count=0,
        )

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        raise AssertionError("cached evaluation should not call predict_with_llm")

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        raise AssertionError("cached evaluation should not call judge")


def test_evaluate_command_uses_fixture_backed_qrel_workflow(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text(
        'method: "custom"\n'
        'system_message: ""\n'
        'prefix_user: "{examples}Query: {query}\\nPassage: {passage}\\n"\n',
        encoding="utf-8",
    )

    qrel_path = FIXTURE_DIR / "sample.qrels"
    run_path = FIXTURE_DIR / "sample.run"
    passage_text = {
        "d0": "passage zero",
        "d1": "passage one",
        "d2": "passage two",
        "d3": "passage three",
    }
    scripted_outputs = [
        "unparseable",
        "0",
        "0",
        "1",
        "1",
        "still not parseable",
        "2",
        "2",
        "2",
        "3",
        "0",
        "3",
    ]

    monkeypatch.setattr(
        qrel_utils,
        "get_query_mappings",
        lambda _: {1: {"title": "fixture query"}},
    )
    monkeypatch.setattr(qrel_utils, "get_qrels_file", lambda qrel: str(Path(qrel)))
    monkeypatch.setattr(
        qrel_utils,
        "get_passage_wrapper",
        lambda _qrel, doc_id: passage_text[str(doc_id)],
    )
    monkeypatch.setattr(
        qrel_utils,
        "fetch_ndcg_score",
        lambda qrel_info, _result_path: (
            "0.1111" if qrel_info == str(qrel_path) else "0.2222"
        ),
    )
    monkeypatch.setattr(
        operations,
        "create_judge",
        lambda args: ScriptedJudge(args.qrel, prompt_file, scripted_outputs),
    )

    exit_code = main(
        [
            "evaluate",
            "--backend",
            "gpt",
            "--model",
            "fixture/model",
            "--qrel",
            str(qrel_path),
            "--result-file",
            str(run_path),
            "--num-sample",
            "3",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "evaluate"
    assert output["metrics"] == {
        "original_ndcg@10": "0.1111",
        "modified_ndcg@10": "0.2222",
    }
    assert output["resolved"]["num_sample"] == 3

    artifact_paths = [artifact["path"] for artifact in output["artifacts"]]
    assert len(artifact_paths) == len(set(artifact_paths))

    result_path = next(
        artifact["path"]
        for artifact in output["artifacts"]
        if artifact["name"] == "evaluation-output"
    )
    assert Path(result_path).exists()
    assert qrel_utils.get_qrels(result_path) == {
        1: {"d0": "0", "d1": "1", "d2": "2", "d3": "3"}
    }
    assert any(path.endswith(".png") for path in artifact_paths)
    assert output["warnings"]


def test_evaluate_command_reuses_existing_modified_qrel_without_rerunning_judge(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text(
        'method: "custom"\n'
        'system_message: ""\n'
        'prefix_user: "{examples}Query: {query}\\nPassage: {passage}\\n"\n',
        encoding="utf-8",
    )

    qrel_path = FIXTURE_DIR / "sample.qrels"
    run_path = FIXTURE_DIR / "sample.run"
    modified_dir = tmp_path / "modified_qrels"
    modified_dir.mkdir()
    cached_result = modified_dir / "sample.q_model_0123_0_1.txt"
    cached_result.write_text(
        "1 0 d0 0\n1 0 d1 1\n1 0 d2 2\n1 0 d3 3\n", encoding="utf-8"
    )

    monkeypatch.setattr(qrel_utils, "get_qrels_file", lambda qrel: str(Path(qrel)))
    monkeypatch.setattr(
        qrel_utils,
        "fetch_ndcg_score",
        lambda qrel_info, _result_path: (
            "0.1111" if qrel_info == str(qrel_path) else "0.3333"
        ),
    )
    monkeypatch.setattr(
        operations,
        "create_judge",
        lambda args: NoJudgeReuse(args.qrel, prompt_file),
    )

    exit_code = main(
        [
            "evaluate",
            "--backend",
            "gpt",
            "--model",
            "fixture/model",
            "--qrel",
            str(qrel_path),
            "--result-file",
            str(run_path),
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "evaluate"
    assert output["metrics"] == {
        "original_ndcg@10": "0.1111",
        "modified_ndcg@10": "0.3333",
    }

    artifact_paths = [artifact["path"] for artifact in output["artifacts"]]
    assert "modified_qrels/sample.q_model_0123_0_1.txt" in artifact_paths
    assert any(path.endswith(".png") for path in artifact_paths)


def test_ensemble_evaluate_command_combines_votes_and_deduplicates_artifacts(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    qrel_path = FIXTURE_DIR / "sample.qrels"
    run_path = FIXTURE_DIR / "sample.run"
    shared_artifact = tmp_path / "shared.txt"
    shared_artifact.write_text("shared", encoding="utf-8")
    result_one = tmp_path / "judge_one.txt"
    result_one.write_text("1 0 d0 2\n1 0 d1 1\n1 0 d2 2\n1 0 d3 3\n", encoding="utf-8")
    result_two = tmp_path / "judge_two.txt"
    result_two.write_text("1 0 d0 1\n1 0 d1 1\n1 0 d2 3\n1 0 d3 3\n", encoding="utf-8")

    results = [
        operations.EvaluateResult(
            result_path=str(result_one),
            stdout="judge-one\n",
            metrics={"original_ndcg@10": "0.1"},
            artifact_paths=[str(result_one), str(shared_artifact)],
        ),
        operations.EvaluateResult(
            result_path=str(result_two),
            stdout="judge-two\n",
            metrics={"original_ndcg@10": "0.2"},
            artifact_paths=[str(result_two), str(shared_artifact)],
        ),
    ]

    def fake_single_judge(args: Any) -> operations.EvaluateResult:
        del args
        return results.pop(0)

    monkeypatch.setattr(operations, "_evaluate_with_single_judge", fake_single_judge)
    monkeypatch.setattr(qrel_utils, "get_qrels_file", lambda qrel: str(Path(qrel)))

    exit_code = main(
        [
            "evaluate",
            "--backend",
            "ensemble",
            "--qrel",
            str(qrel_path),
            "--result-file",
            str(run_path),
            "--llm-judges",
            "GPTJudge,HGFLLMJudge",
            "--model-names",
            "fixture/model-a,fixture/model-b",
            "--output",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["command"] == "evaluate"
    assert output["metrics"] == {}

    artifact_paths = [artifact["path"] for artifact in output["artifacts"]]
    assert artifact_paths.count(str(shared_artifact)) == 1
    final_result = Path(
        next(
            artifact["path"]
            for artifact in output["artifacts"]
            if artifact["name"] == "evaluation-output"
        )
    )
    assert qrel_utils.get_qrels(str(final_result)) == {
        1: {"d0": "1", "d1": "1", "d2": "2", "d3": "3"}
    }
    assert output["warnings"] == ["judge-one\njudge-two\n"]
