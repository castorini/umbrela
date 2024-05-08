import argparse
from typing_extensions import Optional

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
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 2,
        device: str = "cuda",
        num_gpus: int = 1,
    ) -> None:
        super().__init__(qrel, prompt_file, prompt_type, model_name, few_shot_count)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        self.num_gpus = num_gpus

    def predict_with_llm(self, request_dict, max_new_tokens, prepocess, batch_size=1):
        if prepocess:
            self.query_passage = common_utils.preprocess_request_dict(request_dict)
        else:
            self.query_passage = request_dict
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
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

    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        outputs = self.predict_with_llm(request_dict, max_new_tokens, prepocess)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )


def main():
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

    args = parser.parse_args()

    judge = OSLLMJudge(
        args.qrel, args.model, args.prompt_file, args.prompt_type, args.few_shot_count
    )
    judge.evalute_results_with_qrel(args.result_file)


if __name__ == "__main__":
    main()
