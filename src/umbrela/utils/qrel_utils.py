import json
import random

from pyserini.index.lucene import IndexReader
from pyserini.search import get_qrels, get_topics


def get_catwise_data(qrel_data, few_shot_count):
    examples = []
    for cat in [0, 1, 2, 3]:
        req_tuple_list = []

        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    req_tuple_list.append((qid, doc_id))
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")

        assert (
            len(req_tuple_list) >= few_shot_count
        ), f"Count of judgments available for category {cat} is lesser than {few_shot_count}."

        if len(req_tuple_list):
            samples_for_examples = random.sample(req_tuple_list, few_shot_count)
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
        query = query_mappings[int(example[0])]["title"]
        if qrel in ["dl19-passage", "dl20-passage"]:
            passage = json.loads(index_reader.doc_raw(str(example[1]))).get(
                "contents", ""
            )
        else:
            passage = get_passage(example[1])

        res_json = {"O": int(qrel_data[example[0]][example[1]])}
        prompt_examples += f"""
                ###

                Query: {query}
                Passage: {passage}
                Output: {res_json}
                """
    return prompt_examples


def get_query_mappings(qrel):
    # Query mappings
    topic_mapping = {
        "dl19-passage": "dl19-passage",
        "dl20-passage": "dl20",
        "dl21-passage": "dl21",
        "dl22-passage": "dl22",
        "dl23-passage": "dl23",
    }
    if qrel not in topic_mapping:
        raise ValueError(f"Invalid value for qrel: {qrel}")
    query_mappings = get_topics(topic_mapping[qrel])
    return query_mappings


def get_index_reader(qrel):
    # Index reader
    if qrel in ["dl19-passage", "dl20-passage"]:
        index_reader = IndexReader.from_prebuilt_index("msmarco-v1-passage")
    else:
        index_reader = None
    return index_reader


def generate_examples_prompt(qrel, few_shot_count):
    qrel_data = get_qrels(qrel)
    few_shot_examples = get_catwise_data(qrel_data, few_shot_count)
    query_mappings = get_query_mappings(qrel)
    index_reader = get_index_reader(qrel)
    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, index_reader, qrel, qrel_data
    )
    return prompt_examples
