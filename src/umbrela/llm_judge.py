import asyncio
from abc import ABC, abstractmethod
from typing import Any

from umbrela.evaluation import evaluate_judge_results
from umbrela.prompts import (
    PromptTemplate,
    display_prompt_template,
    get_prompt_template,
    render_prompts,
)
from umbrela.utils import common_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str | None,
        prompt_type: str | None,
        few_shot_count: int,
    ) -> None:
        assert not (prompt_file and prompt_type), (
            "Both prompt_file and prompt_type passed. Only one mode must be selected!!"
        )

        self.qrel = qrel
        self.few_shot_count = few_shot_count
        self.prompt_file = prompt_file
        self.model_name = model_name
        if few_shot_count > 0:
            from umbrela.utils import qrel_utils

            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel, few_shot_count
            )
        elif few_shot_count == 0:
            self.prompt_examples = ""
            if prompt_file and "fewshot" in prompt_file:
                print(
                    "Warning!! default fewshot prompt file being used for "
                    "few_shot_count = 0"
                )
        else:
            raise ValueError(f"Invalid value for few_shot_count: {few_shot_count}")

        self.prompt_template: PromptTemplate = get_prompt_template(
            prompt_file=prompt_file,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
        )

    def display_prompt_template(self) -> None:
        display_prompt_template(self.prompt_template)

    def prepare_request_inputs(
        self, request_dict: dict[str, Any] | common_utils.QueryPassage, preprocess: bool
    ) -> tuple[common_utils.QueryPassage, list[str]]:
        query_passage = common_utils.prepare_request_inputs(
            request_dict,
            preprocess,
        )
        prompts = render_prompts(
            self.prompt_template, query_passage, self.prompt_examples
        )
        self.query_passage = query_passage
        self.prompts = prompts
        return query_passage, prompts

    def prepare_judgments(self, outputs: list[str]) -> list[common_utils.Judgment]:
        return common_utils.prepare_judgments(
            outputs,
            self.query_passage,
            self.prompts,
            self.model_name,
            getattr(self, "reasoning_outputs", None),
        )

    @abstractmethod
    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]: ...

    async def async_predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.predict_with_llm, request_dict, max_new_tokens, preprocess
        )

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = self.predict_with_llm(request_dict, max_new_tokens, preprocess)
        return self.prepare_judgments(outputs)

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

    def evalute_results_with_qrel(
        self,
        result_file: str | None,
        judge_cat: list[int] | None = None,
        regenerate: bool = False,
        num_samples: int = 1,
        return_results_path: bool = False,
    ) -> str | None:
        return evaluate_judge_results(
            self,
            result_file=result_file,
            judge_cat=judge_cat,
            regenerate=regenerate,
            num_samples=num_samples,
            return_results_path=return_results_path,
        )
