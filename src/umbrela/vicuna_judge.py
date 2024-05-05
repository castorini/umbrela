from typing_extensions import Optional

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils


class VicunaJudge(LLMJudge):
    def __init__(
        self,
        qrel: str,
        prompt_file: Optional[str] = None,
        prompt_type: Optional[str] = "bing",
        few_shot_count: int = 2,
        device: str = "cuda",
        num_gpus: int = 1,
    ) -> None:
        super().__init__(qrel, prompt_file, prompt_type, "lmsys/vicuna-7b-v1.5-16k", few_shot_count)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        self.num_gpus = num_gpus

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
        do_sample: bool = True,
        top_p: float = 1.0,
        num_beams: int = 1,
        batch_size: int = 1,
        num_workers: int = 16,
    ):
        self.query_passage = common_utils.preprocess_request_dict(request_dict)
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self._prompt_template
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        tokenizer.use_default_system_prompt = False
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        model.eval()

        dataset = datasets.Dataset.from_list([{"text": (t)} for t in self.prompts])

        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        )

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        test_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=DataCollatorWithPadding(tokenizer, padding="longest"),
        )

        outputs = []
        for batch in tqdm(test_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(self._device)

            batch_size, seq_length = batch["input_ids"].shape

            with torch.no_grad():
                output = model.generate(
                    **batch,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    num_beams=num_beams,
                )

            for b in range(batch_size):
                if model.config.is_encoder_decoder:
                    output_ids = output[b]
                else:
                    output_ids = output[b, seq_length:]

                outputs.append(
                    tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                )

        return outputs

    def judge(self, request_dict, max_new_tokens=100):
        outputs = self.predict_with_llm(request_dict, max_new_tokens)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )
