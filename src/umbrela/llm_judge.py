from abc import ABC, abstractmethod
import pkg_resources
import os

from umbrela.utils import qrel_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel,
        prompt_file,
        prompt_type,
        model_name,
        few_shot_count,
    ) -> None:
        assert not (
            prompt_file and prompt_type
        ), "Both prompt_file and prompt_type passed. Only one mode must be selected!!"

        if prompt_type:
            if prompt_type not in ["bing", "basic"]:
                raise ValueError(f"Invalid prompt_type: {prompt_type}.")
            prompt_mode_str = "fewshot" if few_shot_count > 0 else "zeroshot"
            prompt_file = pkg_resources.resource_filename(
                "umbrela", f"prompts/qrel_{prompt_mode_str}_{prompt_type}.txt"
            )
            if not os.path.exists(prompt_file):
                raise ValueError(f"Prompt file doesn't exist.")

        if prompt_file:
            print(
                "Warning!! Prompt file expects input fields namely: (examples, query, passage)."
            )
        self.model_name = model_name
        if few_shot_count > 0:
            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel, few_shot_count
            )
        elif few_shot_count == 0:
            self.prompt_examples = ""
            if "fewshot" in prompt_file:
                print(
                    f"Warning!! default fewshot prompt file being used for few_shot_count = 0"
                )
        else:
            raise ValueError(f"Invalid value for few_shot_count: {few_shot_count}")

        with open(prompt_file) as p:
            self._prompt_template = "".join(p.readlines()).strip()

    def display_prompt_template(self):
        print(self._prompt_template)

    @abstractmethod
    def predict_with_llm(self, request_dict, max_new_tokens):
        pass

    @abstractmethod
    def judge(self, request_dict, max_new_tokens):
        pass
