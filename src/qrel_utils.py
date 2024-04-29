import json
import random

import pandas as pd
from pyserini.index.lucene import IndexReader
from pyserini.search import get_qrels


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


def get_passage(pid):
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(
        f"../data/msmarco_v2_passage/msmarco_passage_{bundlenum}", "rt", encoding="utf8"
    ) as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid
        return document["passage"]


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


def get_query_mappings(qrel):
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
    return query_mappings


def get_index_reader(qrel):
    # Index reader
    if qrel in ["dl19-passage", "dl20-passage"]:
        index_reader = IndexReader("indexes/lucene-index-msmarco-passage")
    else:
        index_reader = None
    return index_reader


def generate_examples_prompt(qrel):
    qrel_data = get_qrels(qrel)
    few_shot_examples = get_catwise_data(qrel_data)
    query_mappings = get_query_mappings(qrel)
    index_reader = get_index_reader(qrel)
    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, index_reader, qrel, qrel_data
    )
    return prompt_examples
