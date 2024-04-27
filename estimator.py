import argparse
import json
import os
import random
import re

import datasets
from dotenv import load_dotenv
import openai
from openai import AzureOpenAI
import pandas as pd
from pyserini.search import get_qrels, get_qrels_file
from pyserini.index.lucene import IndexReader
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm

load_dotenv()

# For GPT connection
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_API_BASE = os.environ["AZURE_OPENAI_API_BASE"]
OPEN_AI_API_KEY = os.environ["OPEN_AI_API_KEY"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]


def get_passage(pid):
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(
        f"./msmarco_v2_passage/msmarco_passage_{bundlenum}", "rt", encoding="utf8"
    ) as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid
        return document["passage"]


def get_catwise_data(qrel_data):
    examples = []
    for cat in [0, 1, 2, 3]:
        req_tuple_list = []

        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    req_tuple_list.append((qid, doc_id))
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")

        assert (
            len(req_tuple_list) >= 2
        ), f"Count of judgments available for category {cat} is lesser than 2."

        if len(req_tuple_list):
            samples_for_examples = random.sample(req_tuple_list, 2)
            examples.extend(samples_for_examples)
    return examples


def parse_fewshot_response(response: str, passage: str, query: str) -> int:
    response = response.strip().lower()
    valid_res = 1
    answer = ""
    patterns = [
        r'"o": (0|1|2|3)',
        r'"overall_score": (0|1|2|3)',
        r'"overall": (0|1|2|3)',
        r'"overall score": (0|1|2|3)',
        r'"final score": (0|1|2|3)',
        r'"final_score": (0|1|2|3)',
        r'"score": (0|1|2|3)',
        r'"o_score": (0|1|2|3)',
    ]
    for pattern in patterns:
        matched = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)

        if matched:
            answer = matched.group(1).capitalize()
            break
    if answer == "":
        answer = "0"
        valid_res = 0
        print(f"Invalid response to `{query}` & `{passage}`: {response}")
    return int(answer), valid_res


def examples_prompt(few_shot_examples, query_mappings, index_reader, qrel, qrel_data):
    prompt_examples = ""

    for example in few_shot_examples:
        query = query_mappings[int(example[0])]
        if qrel in ["dl19-passage", "dl20-passage"]:
            passage = json.loads(index_reader.doc_raw(str(example[1]))).get(
                "contents", ""
            )
        else:
            passage = get_passage(example[1])

        prompt_examples += f"""
                ###

                Query: {query}
                Passage: {passage}
                Relevance category: {qrel_data[example[0]][example[1]]}
                """
    return prompt_examples


def run_gpt(curr_prompt, client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": curr_prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME, messages=messages, max_tokens=130
        )
        output = (
            response.choices[0].message.content.lower()
            if response.choices[0].message.content
            else ""
        )
    except openai.BadRequestError as e:
        print(f"Encountered {e} for {curr_prompt}")
        output = ""

    return output


def run_inference_batch(
    texts: list,
    model_name,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    top_p: float = 1.0,
    num_beams: int = 1,
    batch_size: int = 1,
    num_workers: int = 16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.use_default_system_prompt = False
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    model.eval()

    dataset = datasets.Dataset.from_list([{"text": (t)} for t in texts])

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
            batch[key] = batch[key].to("cuda")

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


def eval(qrel, result_file, prompt_file, model_name, modified_qrel):

    if "gpt" in model_name:
        client = AzureOpenAI(
            api_key=OPEN_AI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_API_BASE,
        )

    # Query mappings
    mapping_file = {
        "dl19-passage": "query_mappings/2019_queries.tsv",
        "dl20-passage": "query_mappings/2020_queries.tsv",
        "dl21-passage": "query_mappings/2021_queries.tsv",
        "dl22-passage": "query_mappings/2022_queries.tsv",
        "dl23-passage": "query_mappings/2023_queries.tsv",
    }
    query_mappings_df = pd.read_csv(
        mapping_file[qrel], sep="\t", names=["qid", "query"]
    )
    query_mappings = dict(zip(query_mappings_df["qid"], query_mappings_df["query"]))

    # Index reader
    if qrel in ["dl19-passage", "dl20-passage"]:
        index_reader = IndexReader("indexes/lucene-index-msmarco-passage")
    else:
        index_reader = None

    qrel_data = get_qrels(qrel)

    few_shot_examples = get_catwise_data(qrel_data=qrel_data)

    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, index_reader, qrel, qrel_data
    )

    with open(prompt_file) as p:
        prompt_template = "".join(p.readlines()).strip()

    with open(result_file, "rb") as f_in:
        lines = [line for line in f_in]

    counter = 0
    holes = []
    for line in lines:
        counter += 1
        decoded = line.decode("utf-8").split(" ")
        qid = int(decoded[0])
        doc_id = decoded[2]
        if not qrel_data.get(qid, None) or not qrel_data[qid].get(doc_id, None):
            holes.append((qid, doc_id))
    print(f"Count of holes: {len(holes)} out of {counter}")

    prompts = []
    valid_res_count = 0
    for sample in holes:
        if qrel in ["dl19-passage", "dl20-passage"]:
            passage = json.loads(index_reader.doc_raw(str(sample[1]))).get(
                "contents", ""
            )
        else:
            passage = get_passage(sample[1])
        curr_prompt = prompt_template.format(
            examples=prompt_examples,
            query=query_mappings[int(sample[0])],
            passage=passage,
        )
        prompts.append(curr_prompt)

    if "gpt" in model_name:
        outputs = [run_gpt(prompt, client, query_mappings) for prompt in tqdm(prompts)]
    elif "vicuna" in model_name:
        outputs = run_inference_batch(prompts, model_name)

    for response, sample in zip(outputs, holes):
        result, valid_res = parse_fewshot_response(
            response, passage, query_mappings[int(sample[0])]
        )
        valid_res_count += valid_res
        if int(sample[0]) in qrel_data:
            qrel_data[int(sample[0])][sample[1]] = result
        else:
            qrel_data[int(sample[0])] = {sample[1]: result}


    with open(modified_qrel, "wb") as f_out:
        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                result = str(qrel_data[qid][doc_id]) + "\n"
                encoded = " ".join([str(qid), "0", doc_id, result]).encode("utf-8")
                f_out.write(encoded)

    cmd1 = (
        f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {qrel} {result_file}"
    )
    os.system(cmd1)
    cmd2 = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {modified_qrel} {result_file}"
    os.system(cmd2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file", required=True)
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument(
        "--prompt_file", type=str, help="prompt file", default="qrel_fewshot_bing.txt"
    )
    parser.add_argument("--model", type=str, help="model name")

    args = parser.parse_args()

    os.makedirs("modified_qrels", exist_ok=True)

    rename_dict = {"lmsys/vicuna-7b-v1.5-16k": "vicuna", "gpt": "gpt"}
    result_dir = f"modified_qrels/{rename_dict[args.model]}"
    os.makedirs(result_dir, exist_ok=True)

    path = get_qrels_file(args.qrel)
    output_filename = f"{result_dir}/{os.path.basename(path)[:-4]}_{args.prompt_file}"

    print(f"Output file: {output_filename}")

    eval(args.qrel, args.result_file, args.prompt_file, args.model, output_filename)


if __name__ == "__main__":
    main()
