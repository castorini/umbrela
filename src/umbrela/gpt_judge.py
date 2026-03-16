import asyncio
import os
from typing import Any

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_REASONING_MAX_NEW_TOKENS = 4096


class GPTJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str | None = None,
        prompt_type: str | None = "bing",
        few_shot_count: int = 0,
        use_azure_openai: bool = False,
        use_openrouter: bool = False,
        max_concurrency: int = 8,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.max_concurrency = max_concurrency
        self.reasoning_effort = reasoning_effort
        self.use_openrouter = use_openrouter
        self.create_openai_client(
            use_azure_openai=use_azure_openai,
            use_openrouter=use_openrouter,
        )

    def _uses_reasoning_style_api(self) -> bool:
        return (
            "o1" in self.model_name
            or "o3" in self.model_name
            or "o4" in self.model_name
            or "gpt-5" in self.model_name
        )

    def _resolve_max_new_tokens(self, max_new_tokens: int) -> int:
        if (
            self._uses_reasoning_style_api()
            and max_new_tokens == DEFAULT_MAX_NEW_TOKENS
        ):
            return DEFAULT_REASONING_MAX_NEW_TOKENS
        return max_new_tokens

    def create_openai_client(
        self, use_azure_openai: bool = False, use_openrouter: bool = False
    ) -> None:
        try:
            import openai
            from openai import AsyncAzureOpenAI, AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "GPTJudge requires the OpenAI SDK. Install umbrela with "
                "`uv sync --extra cloud`."
            ) from exc

        openai_api_key = os.getenv("OPENAI_API_KEY") or None
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or None
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or openai_api_key
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
        self._bad_request_error = openai.BadRequestError
        self.async_client: Any

        if use_azure_openai and use_openrouter:
            raise ValueError(
                "OpenRouter and Azure OpenAI are mutually exclusive provider "
                "flags. Set only one of `use_azure_openai` or `use_openrouter`."
            )

        if use_azure_openai:
            if not all([azure_api_key, azure_endpoint, api_version]):
                raise ValueError(
                    "Azure OpenAI requested but one or more required environment "
                    "variables are missing: `AZURE_OPENAI_API_BASE`, "
                    "`AZURE_OPENAI_API_VERSION`, and "
                    "`AZURE_OPENAI_API_KEY` (or `OPENAI_API_KEY` as fallback)."
                )
            assert azure_endpoint is not None
            self.async_client = AsyncAzureOpenAI(
                api_key=azure_api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            self.use_azure_ai = True
            self.use_openrouter = False
            self.engine = os.environ["DEPLOYMENT_NAME"]
        else:
            provider_api_key = openai_api_key
            client_kwargs: dict[str, Any] = {}

            if use_openrouter:
                if openrouter_api_key is None:
                    raise KeyError("OPENROUTER_API_KEY")
                provider_api_key = openrouter_api_key
                client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            elif provider_api_key is None and openrouter_api_key is not None:
                provider_api_key = openrouter_api_key
                client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
                self.use_openrouter = True

            if provider_api_key is None:
                raise KeyError("OPENAI_API_KEY or OPENROUTER_API_KEY")

            self.async_client = AsyncOpenAI(
                api_key=provider_api_key,
                **client_kwargs,
            )
            self.engine = self.model_name
            self.use_azure_ai = False

    def _normalize_messages(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if (
            "o1" in self.model_name
            or "o3" in self.model_name
            or "o4" in self.model_name
        ):
            normalized_messages = [message.copy() for message in messages[1:]]
            normalized_messages[0]["content"] = (
                messages[0]["content"] + "\n" + messages[1]["content"]
            )
            return normalized_messages
        return messages

    def _build_completion_params(
        self, messages: list[dict[str, str]], max_new_tokens: int
    ) -> dict[str, Any]:
        max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)
        normalized_messages = self._normalize_messages(messages)
        uses_reasoning_style_api = self._uses_reasoning_style_api()
        temperature = 0.0
        if uses_reasoning_style_api:
            temperature = 1.0

        completion_params: dict[str, Any] = {
            "model": self.engine,
            "messages": normalized_messages,
            "temperature": temperature,
            "timeout": 30,
        }
        if uses_reasoning_style_api:
            completion_params["max_completion_tokens"] = max_new_tokens
        else:
            completion_params["max_tokens"] = max_new_tokens
            completion_params["top_p"] = 1
            completion_params["frequency_penalty"] = 0.5
            completion_params["presence_penalty"] = 0
        return completion_params

    def _build_responses_params(
        self, messages: list[dict[str, str]], max_new_tokens: int
    ) -> dict[str, Any]:
        max_new_tokens = self._resolve_max_new_tokens(max_new_tokens)
        return {
            "model": self.engine,
            "input": [
                {
                    "type": "message",
                    "role": message["role"],
                    "content": [
                        {
                            "type": "input_text",
                            "text": message["content"],
                        }
                    ],
                }
                for message in self._normalize_messages(messages)
            ],
            "max_output_tokens": max_new_tokens,
            "timeout": 30,
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "auto",
            },
        }

    def _extract_responses_text_and_reasoning(
        self, response: Any
    ) -> tuple[str, str | None]:
        text = ""
        if hasattr(response, "output_text") and response.output_text:
            text = str(response.output_text)
        else:
            for item in getattr(response, "output", []):
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            text = getattr(content, "text", "") or text

        reasoning = None
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "reasoning":
                direct_reasoning_parts: list[str] = []
                direct_reasoning = getattr(item, "reasoning", None)
                if direct_reasoning:
                    direct_reasoning_parts.append(str(direct_reasoning))
                direct_reasoning_content = getattr(item, "reasoning_content", None)
                if direct_reasoning_content:
                    direct_reasoning_parts.append(str(direct_reasoning_content))
                content_items = getattr(item, "content", None) or []
                direct_reasoning_parts.extend(
                    [
                        str(content) if isinstance(content, str) else str(content.text)
                        for content in content_items
                        if (isinstance(content, str) and content)
                        or (hasattr(content, "text") and content.text)
                    ]
                )
                summaries = getattr(item, "summary", None)
                summary_reasoning = None
                if summaries:
                    summary_reasoning = "\n".join(
                        str(summary) if isinstance(summary, str) else summary.text
                        for summary in summaries
                        if (isinstance(summary, str) and summary)
                        or (hasattr(summary, "text") and summary.text)
                    )
                direct_reasoning_text = (
                    "\n".join(direct_reasoning_parts)
                    if direct_reasoning_parts
                    else None
                )
                if self.use_openrouter:
                    reasoning = direct_reasoning_text or summary_reasoning
                else:
                    reasoning = summary_reasoning or direct_reasoning_text
        return text.lower(), reasoning

    async def _run_gpt_with_responses_api(
        self, messages: list[dict[str, str]], max_new_tokens: int
    ) -> tuple[str, str | None]:
        response = await self.async_client.responses.create(
            **self._build_responses_params(messages, max_new_tokens)
        )
        return self._extract_responses_text_and_reasoning(response)

    async def run_gpt(self, prompt: str, max_new_tokens: int) -> tuple[str, str | None]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            try:
                if (
                    self.reasoning_effort is not None
                    and self._uses_reasoning_style_api()
                ):
                    return await self._run_gpt_with_responses_api(
                        messages, max_new_tokens
                    )
                response = await self.async_client.chat.completions.create(
                    **self._build_completion_params(messages, max_new_tokens)
                )
                output = (
                    response.choices[0].message.content.lower()
                    if response.choices[0].message.content
                    else ""
                )
                reasoning = common_utils.extract_reasoning_content(
                    response.choices[0].message
                )
                return output, reasoning
            except self._bad_request_error as e:
                print(f"Encountered {e} for {prompt}")
                return "", None
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.1)

        return "", None

    async def async_predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        _, prompts = self.prepare_request_inputs(request_dict, preprocess)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_prompt(prompt: str) -> tuple[str, str | None]:
            async with semaphore:
                return await self.run_gpt(prompt, max_new_tokens)

        responses = await asyncio.gather(*(run_prompt(prompt) for prompt in prompts))
        self.reasoning_outputs = [reasoning for _, reasoning in responses]
        return [output for output, _ in responses]

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        return common_utils.run_async_blocking(
            self.async_judge(request_dict, max_new_tokens, preprocess)
        )

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        return common_utils.run_async_blocking(
            self.async_predict_with_llm(request_dict, max_new_tokens, preprocess)
        )

    async def async_judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = await self.async_predict_with_llm(
            request_dict, max_new_tokens, preprocess
        )
        return self.prepare_judgments(outputs)
