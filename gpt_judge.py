import os

import openai
from openai import AzureOpenAI
from tqdm import tqdm

import qrel_utils
import utils


class GPTJudge:
    def __init__(self, qrel, prompt_file) -> None:
        self.model_name = "gpt"
        self.client = AzureOpenAI(
            api_key=os.environ["OPEN_AI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
        )
        self.prompt_examples = qrel_utils.generate_examples_prompt(qrel=qrel)
        with open(prompt_file) as p:
            self.prompt_template = "".join(p.readlines()).strip()

    def run_gpt(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=os.environ["DEPLOYMENT_NAME"], messages=messages, max_tokens=100
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

    def predict(
        self,
        request_dict: list,
    ):
        self.query_passage = utils.preprocess_request_dict(request_dict)
        self.prompts = utils.generate_prompts(
            self.query_passage, self.prompt_examples, self.prompt_template
        )

        outputs = [self.run_gpt(prompt) for prompt in tqdm(self.prompts)]
        return outputs

    def judge(self, request_dict):
        outputs = self.predict(request_dict)
        return utils.prepare_judgments(
            outputs, self.query_passage, self.prompts, self.model_name
        )
