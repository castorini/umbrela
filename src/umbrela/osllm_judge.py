import pkg_resources

from fastchat.model import load_model
import torch
from tqdm import tqdm
from transformers.generation import GenerationConfig

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils


class OSLLMJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str = pkg_resources.resource_filename(
            "umbrela", "prompts/qrel_fewshot_bing.txt"
        ),
        few_shot_count: int = 2,
        device: str = "cuda",
        num_gpus: int = 1,
    ) -> None:
        super().__init__(qrel, prompt_file, model_name, few_shot_count)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        self.num_gpus = num_gpus

    def predict_with_llm(self, request_dict, max_new_tokens, batch_size=1):
        self.query_passage = common_utils.preprocess_request_dict(request_dict)
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self.prompt_template
        )

        self._llm, self._tokenizer = load_model(
            self.model_name, device=self._device, num_gpus=self.num_gpus
        )

        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.do_sample = False

        outputs = []
        for i in tqdm(range(0, len(self.prompts), batch_size)):
            inputs = self._tokenizer(self.prompts[i : i + batch_size])
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

    def judge(self, request_dict, max_new_tokens=100):
        outputs = self.predict_with_llm(request_dict, max_new_tokens)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )
