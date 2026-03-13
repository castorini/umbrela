import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from umbrela.gpt_judge import GPTJudge
from umbrela.utils import common_utils


SAMPLE_REQUEST = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {"doc": {"segment": "first passage"}, "docid": "d1"},
        {"doc": {"segment": "second passage"}, "docid": "d2"},
        {"doc": {"segment": "third passage"}, "docid": "d3"},
    ],
}


class GPTJudgeAsyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_file = tempfile.NamedTemporaryFile("w", delete=False)
        self.prompt_file.write("{examples}\nQuery: {query}\nPassage: {passage}\n")
        self.prompt_file.flush()
        self.addCleanup(self.prompt_file.close)
        self.addCleanup(lambda: os.unlink(self.prompt_file.name))

    def make_judge(self) -> GPTJudge:
        def fake_create_openai_client(self, use_azure_openai: bool = False) -> None:
            self.async_client = object()
            self.engine = self.model_name
            self.use_azure_ai = use_azure_openai

        with patch.object(GPTJudge, "create_openai_client", fake_create_openai_client):
            return GPTJudge(
                qrel="dl19-passage",
                model_name="gpt-4o",
                prompt_file=self.prompt_file.name,
                prompt_type=None,
                few_shot_count=0,
                max_concurrency=2,
            )

    def test_async_judge_preserves_input_order(self) -> None:
        judge = self.make_judge()

        async def fake_run_gpt(prompt: str, max_new_tokens: int) -> tuple[str, str | None]:
            if "first passage" in prompt:
                await asyncio.sleep(0.03)
                return "2", "first reasoning"
            if "second passage" in prompt:
                await asyncio.sleep(0.01)
                return "1", None
            await asyncio.sleep(0.02)
            return "3", "third reasoning"

        judge.run_gpt = fake_run_gpt  # type: ignore[method-assign]
        judgments = asyncio.run(judge.async_judge(SAMPLE_REQUEST))

        self.assertEqual([item["passage"] for item in judgments], [
            "first passage",
            "second passage",
            "third passage",
        ])
        self.assertEqual([item["judgment"] for item in judgments], [2, 1, 3])
        self.assertEqual(
            [item["reasoning"] for item in judgments],
            ["first reasoning", None, "third reasoning"],
        )
        self.assertTrue(all("result_status" in item for item in judgments))

    def test_sync_judge_uses_async_path(self) -> None:
        judge = self.make_judge()

        async def fake_async_predict_with_llm(
            request_dict, max_new_tokens: int, prepocess: bool
        ):
            judge.prepare_request_inputs(request_dict, prepocess)
            return ["0", "1", "2"]

        judge.async_predict_with_llm = fake_async_predict_with_llm  # type: ignore[method-assign]
        judgments = judge.judge(SAMPLE_REQUEST)

        self.assertEqual([item["judgment"] for item in judgments], [0, 1, 2])
        self.assertEqual(judgments[1]["passage"], "second passage")
        self.assertIsNone(judgments[1]["reasoning"])

    def test_gpt5_uses_max_completion_tokens(self) -> None:
        judge = self.make_judge()
        judge.model_name = "gpt-5.4"
        judge.engine = "gpt-5.4"

        params = judge._build_completion_params(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            max_new_tokens=123,
        )

        self.assertEqual(params["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", params)
        self.assertEqual(params["temperature"], 1.0)

    def test_o1_models_fold_system_message(self) -> None:
        judge = self.make_judge()
        judge.model_name = "o1-preview"
        judge.engine = "o1-preview"

        params = judge._build_completion_params(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ],
            max_new_tokens=50,
        )

        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["content"], "sys\nusr")
        self.assertEqual(params["max_completion_tokens"], 50)
        self.assertNotIn("max_tokens", params)
        self.assertEqual(params["temperature"], 1.0)

    def test_extract_reasoning_content_handles_message_shapes(self) -> None:
        self.assertEqual(
            common_utils.extract_reasoning_content({"reasoning": "chain"}),
            "chain",
        )
        self.assertEqual(
            common_utils.extract_reasoning_content({"reasoning_content": "scratch"}),
            "scratch",
        )


if __name__ == "__main__":
    unittest.main()
