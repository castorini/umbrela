import os
from typing import Any

import vertexai
from retry import retry
from tqdm import tqdm
from vertexai.generative_models import GenerationConfig, GenerativeModel

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils


class GeminiJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str | None = None,
        prompt_type: str | None = "bing",
        few_shot_count: int = 0,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self.create_gemini_client()

    def create_gemini_client(self) -> None:
        vertexai.init(
            project=os.environ["GCLOUD_PROJECT"], location=os.environ["GCLOUD_REGION"]
        )
        self.client = GenerativeModel(self.model_name)

    @retry(tries=3, delay=0.1)
    def run_gemini(self, prompt: str, max_new_tokens: int) -> str:
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    max_output_tokens=max_new_tokens,
                ),
            )
            output = getattr(response, "text", "")
        except Exception:
            output = ""
        return str(output)

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        _, prompts = self.prepare_request_inputs(request_dict, preprocess)

        outputs = [self.run_gemini(prompt, max_new_tokens) for prompt in tqdm(prompts)]
        return outputs
