import os

import openai
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

from llm_judge import LLMJudge
from src.umbrela.utils import common_utils


class GPTJudge(LLMJudge):
    def __init__(
        self,
        qrel,
        prompt_file,
        few_shot_count=2,
        engine=""
    ) -> None:
        model_name = engine if engine else "gpt"
        super().__init__(qrel, prompt_file, model_name, few_shot_count)
        self.create_openai_client()

    def create_openai_client(self):
        api_key = os.environ["OPEN_AI_API_KEY"]
        api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        azure_endpoint = os.environ["AZURE_OPENAI_API_BASE"]

        if all([api_key, azure_endpoint, api_version]):
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            self.use_azure_ai = True
            self.engine = os.environ["DEPLOYMENT_NAME"]
        else:
            self.client = OpenAI(api_key=api_key)
            self.engine = self.model_name
            self.use_azure_ai = False

    def run_gpt(self, prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=os.environ["DEPLOYMENT_NAME"],
                messages=messages,
                max_tokens=max_new_tokens,
            )
            output = (
                response.choices[0].message.content.lower()
                if response.choices[0].message.content
                else ""
            )
        except openai.BadRequestError as e:
            print(f"Encountered {e} for {prompt}")
            output = ""

        return output

    def predict_with_llm(
        self,
        request_dict: list,
        max_new_tokens: int,
    ):
        self.query_passage = common_utils.preprocess_request_dict(request_dict)
        self.prompts = common_utils.generate_prompts(
            self.query_passage, self.prompt_examples, self.prompt_template
        )

        outputs = [
            self.run_gpt(prompt, max_new_tokens) for prompt in tqdm(self.prompts)
        ]
        return outputs

    def judge(self, request_dict, max_new_tokens=100):
        outputs = self.predict_with_llm(request_dict, max_new_tokens)
        return common_utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )
