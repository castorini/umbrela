from typing import Any

from fastchat.model import load_model
import torch
from tqdm import tqdm
from transformers.generation import GenerationConfig

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class OSLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str | None = None,
        prompt_type: str | None = "bing",
        few_shot_count: int = 2,
        device: str = "cuda",
        num_gpus: int = 1,
    ) -> None:
        super().__init__(qrel, model_name, prompt_file, prompt_type, few_shot_count)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        self.num_gpus = num_gpus

    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        prepocess: bool,
        batch_size: int = 1,
    ) -> list[str]:
        _, prompts = self.prepare_request_inputs(request_dict, prepocess)

        self._llm, self._tokenizer = load_model(
            self.model_name, device=self._device, num_gpus=self.num_gpus
        )

        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.do_sample = False

        outputs: list[str] = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            inputs = self._tokenizer(prompts[i : i + batch_size])
            inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items()}
            output = self._llm.generate(**inputs, generation_config=gen_cfg)
            for b in range(batch_size):
                if self._llm.config.is_encoder_decoder:
                    output_ids = output[b]
                else:
                    output_ids = output[b, len(inputs["input_ids"][b]) :]

                outputs.append(
                    self._tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    ).strip()
                )
        return outputs

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        prepocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return self.prepare_judgments(outputs)
