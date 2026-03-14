import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

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
        self.prompt_file = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        )
        self.prompt_file.write(
            'method: "custom"\n'
            'system_message: ""\n'
            'prefix_user: "{examples}\\nQuery: {query}\\nPassage: {passage}\\n"\n'
        )
        self.prompt_file.flush()
        self.addCleanup(self.prompt_file.close)
        self.addCleanup(lambda: os.unlink(self.prompt_file.name))

    def make_judge(self) -> GPTJudge:
        def fake_create_openai_client(
            self: GPTJudge,
            use_azure_openai: bool = False,
            use_openrouter: bool = False,
        ) -> None:
            self.async_client = cast(
                Any,
                SimpleNamespace(
                    responses=SimpleNamespace(create=AsyncMock()),
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=AsyncMock())
                    ),
                ),
            )
            self.engine = self.model_name
            self.use_azure_ai = use_azure_openai
            self.use_openrouter = use_openrouter

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

        async def fake_run_gpt(
            prompt: str, max_new_tokens: int
        ) -> tuple[str, str | None]:
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

        self.assertEqual(
            [item["passage"] for item in judgments],
            [
                "first passage",
                "second passage",
                "third passage",
            ],
        )
        self.assertEqual([item["judgment"] for item in judgments], [2, 1, 3])
        self.assertEqual(
            [item["reasoning"] for item in judgments],
            ["first reasoning", None, "third reasoning"],
        )
        self.assertTrue(all("result_status" in item for item in judgments))

    def test_sync_judge_uses_async_path(self) -> None:
        judge = self.make_judge()

        async def fake_async_predict_with_llm(
            request_dict: dict[str, Any], max_new_tokens: int, prepocess: bool
        ) -> list[str]:
            del max_new_tokens
            judge.prepare_request_inputs(request_dict, prepocess)
            return ["0", "1", "2"]

        judge.async_predict_with_llm = fake_async_predict_with_llm  # type: ignore[assignment,method-assign]
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

    def test_reasoning_models_default_to_4096_max_tokens(self) -> None:
        judge = self.make_judge()
        judge.model_name = "gpt-5.4"
        judge.engine = "gpt-5.4"

        params = judge._build_completion_params(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            max_new_tokens=100,
        )

        self.assertEqual(params["max_completion_tokens"], 4096)

    def test_non_reasoning_models_keep_default_max_tokens(self) -> None:
        judge = self.make_judge()

        params = judge._build_completion_params(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            max_new_tokens=100,
        )

        self.assertEqual(params["max_tokens"], 100)

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

    def test_reasoning_effort_uses_responses_api(self) -> None:
        judge = self.make_judge()
        judge.model_name = "gpt-5.4"
        judge.engine = "gpt-5.4"
        judge.reasoning_effort = "medium"

        response = SimpleNamespace(
            output_text="3",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=[SimpleNamespace(text="reason summary")],
                )
            ],
        )
        responses_create = cast(AsyncMock, judge.async_client.responses.create)
        chat_create = cast(AsyncMock, judge.async_client.chat.completions.create)
        responses_create.return_value = response

        output, reasoning = asyncio.run(judge.run_gpt("prompt", 77))

        responses_create.assert_awaited_once_with(
            model="gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a helpful assistant.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "prompt"}],
                },
            ],
            max_output_tokens=77,
            timeout=30,
            reasoning={"effort": "medium", "summary": "auto"},
        )
        chat_create.assert_not_called()
        self.assertEqual(output, "3")
        self.assertEqual(reasoning, "reason summary")

    def test_reasoning_effort_defaults_to_4096_max_output_tokens(self) -> None:
        judge = self.make_judge()
        judge.model_name = "gpt-5.4"
        judge.engine = "gpt-5.4"
        judge.reasoning_effort = "medium"

        response = SimpleNamespace(output_text="3", output=[])
        responses_create = cast(AsyncMock, judge.async_client.responses.create)
        responses_create.return_value = response

        asyncio.run(judge.run_gpt("prompt", 100))

        responses_create.assert_awaited_once_with(
            model="gpt-5.4",
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a helpful assistant.",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "prompt"}],
                },
            ],
            max_output_tokens=4096,
            timeout=30,
            reasoning={"effort": "medium", "summary": "auto"},
        )

    def test_openrouter_reasoning_models_also_use_responses_api(self) -> None:
        judge = self.make_judge()
        judge.model_name = "openrouter/openai/o4-mini"
        judge.engine = "openrouter/openai/o4-mini"
        judge.reasoning_effort = "high"
        judge.use_openrouter = True

        response = SimpleNamespace(
            output_text="2",
            output=[SimpleNamespace(type="reasoning", summary=["router summary"])],
        )
        responses_create = cast(AsyncMock, judge.async_client.responses.create)
        chat_create = cast(AsyncMock, judge.async_client.chat.completions.create)
        responses_create.return_value = response

        output, reasoning = asyncio.run(judge.run_gpt("prompt", 77))

        responses_create.assert_awaited_once()
        chat_create.assert_not_called()
        self.assertEqual(output, "2")
        self.assertEqual(reasoning, "router summary")

    def test_openrouter_responses_prefer_direct_reasoning(self) -> None:
        judge = self.make_judge()
        judge.model_name = "openrouter/openai/o4-mini"
        judge.engine = "openrouter/openai/o4-mini"
        judge.reasoning_effort = "high"
        judge.use_openrouter = True

        response = SimpleNamespace(
            output_text="2",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    reasoning="raw router reasoning",
                    summary=["router summary"],
                )
            ],
        )
        responses_create = cast(AsyncMock, judge.async_client.responses.create)
        responses_create.return_value = response

        output, reasoning = asyncio.run(judge.run_gpt("prompt", 77))

        self.assertEqual(output, "2")
        self.assertEqual(reasoning, "raw router reasoning")

    def test_openrouter_mode_uses_openrouter_base_url(self) -> None:
        client_kwargs: dict[str, Any] = {}

        def fake_async_openai(**kwargs: Any) -> Any:
            client_kwargs.update(kwargs)
            return SimpleNamespace(
                responses=SimpleNamespace(create=AsyncMock()),
                chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
            )

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=False),
            patch("openai.AsyncOpenAI", side_effect=fake_async_openai),
        ):
            judge = GPTJudge(
                qrel="dl19-passage",
                model_name="openai/gpt-4o-mini",
                prompt_file=self.prompt_file.name,
                prompt_type=None,
                few_shot_count=0,
                use_openrouter=True,
            )

        self.assertEqual(client_kwargs["api_key"], "router-key")
        self.assertEqual(client_kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertTrue(judge.use_openrouter)

    def test_openrouter_fallback_is_used_when_openai_key_is_missing(self) -> None:
        client_kwargs: dict[str, Any] = {}

        def fake_async_openai(**kwargs: Any) -> Any:
            client_kwargs.update(kwargs)
            return SimpleNamespace(
                responses=SimpleNamespace(create=AsyncMock()),
                chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
            )

        with (
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "router-key", "OPENAI_API_KEY": ""},
                clear=False,
            ),
            patch("openai.AsyncOpenAI", side_effect=fake_async_openai),
        ):
            judge = GPTJudge(
                qrel="dl19-passage",
                model_name="anthropic/claude-3.5-sonnet",
                prompt_file=self.prompt_file.name,
                prompt_type=None,
                few_shot_count=0,
            )

        self.assertEqual(client_kwargs["api_key"], "router-key")
        self.assertEqual(client_kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertTrue(judge.use_openrouter)

    def test_prepare_judgments_falls_back_to_reasoning_for_score(self) -> None:
        judgments = common_utils.prepare_judgments(
            outputs=[""],
            query_passage=[("query text", "passage text")],
            prompts=["prompt"],
            model_name="gpt-5.4",
            reasoning_outputs=["I conclude with a score of 2."],
        )

        self.assertEqual(judgments[0]["judgment"], 2)
        self.assertEqual(judgments[0]["result_status"], 1)


if __name__ == "__main__":
    unittest.main()
