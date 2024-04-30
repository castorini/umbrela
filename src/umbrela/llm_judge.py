from abc import ABC, abstractmethod

from src.umbrela.utils import qrel_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel,
        prompt_file,
        model_name,
        few_shot_count,
    ) -> None:
        self.model_name = model_name
        if few_shot_count > 0:
            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel, few_shot_count
            )
        else:
            self.prompt_examples = ""

        with open(prompt_file) as p:
            self.prompt_template = "".join(p.readlines()).strip()

    @abstractmethod
    def predict_with_llm(self, request_dict, max_new_tokens):
        pass

    @abstractmethod
    def judge(self, request_dict, max_new_tokens):
        pass
