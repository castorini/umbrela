import argparse
import os
from typing import Any

import vertexai
from dotenv import load_dotenv
from retry import retry
from tqdm import tqdm
from vertexai.generative_models import GenerationConfig, GenerativeModel

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


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
            output = response.text
        except Exception:
            output = ""
        return output

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        prepocess: bool,
    ) -> list[str]:
        _, prompts = self.prepare_request_inputs(request_dict, prepocess)

        outputs = [self.run_gemini(prompt, max_new_tokens) for prompt in tqdm(prompts)]
        return outputs

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        prepocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return self.prepare_judgments(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file", required=True)
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument(
        "--prompt_type", type=str, help="Prompt type. Supported types: [bing, basic]."
    )
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    judge = GeminiJudge(
        args.qrel, args.model, args.prompt_file, args.prompt_type, args.few_shot_count
    )
    judge.evalute_results_with_qrel(
        args.result_file,
        regenerate=args.regenerate,
        num_samples=args.num_sample,
        judge_cat=JUDGE_CAT,
    )


if __name__ == "__main__":
    main()
