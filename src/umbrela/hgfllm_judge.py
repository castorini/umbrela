import os
from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from umbrela.llm_judge import LLMJudge
from umbrela.utils import common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


class HGFLLMJudge(LLMJudge):
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
        preprocess: bool,
        do_sample: bool = True,
        top_p: float = 1.0,
        num_beams: int = 1,
        batch_size: int = 1,
        num_workers: int = 16,
    ) -> list[str]:
        _, prompts = self.prepare_request_inputs(request_dict, preprocess)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=os.environ["HF_TOKEN"],
            cache_dir=os.environ["HF_CACHE_DIR"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=os.environ["HF_TOKEN"],
            cache_dir=os.environ["HF_CACHE_DIR"],
        )
        tokenizer.use_default_system_prompt = False
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        model.eval()

        dataset = datasets.Dataset.from_list([{"text": t} for t in prompts])

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

        outputs: list[str] = []
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

    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = self.predict_with_llm(request_dict, max_new_tokens, preprocess)
        return self.prepare_judgments(outputs)
