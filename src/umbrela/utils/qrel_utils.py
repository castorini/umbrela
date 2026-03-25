import json
import os
import platform
import random
import re
import subprocess
from functools import lru_cache
from typing import Any, TypeAlias, cast

try:
    from pyserini.index.lucene import LuceneIndexReader
    from pyserini.search import get_qrels_file, get_topics

    _PYSERINI_IMPORT_ERROR = None
except Exception as exc:
    LuceneIndexReader = None
    get_qrels_file = None
    get_topics = None
    _PYSERINI_IMPORT_ERROR = exc

QrelKey: TypeAlias = int | str
QrelsData: TypeAlias = dict[QrelKey, dict[QrelKey, str]]
QueryMappings: TypeAlias = dict[int, dict[str, str]]


def _require_pyserini(feature: str, needs_java: bool = False) -> None:
    if get_qrels_file is None or get_topics is None:
        message = (
            f"{feature} requires the optional `pyserini` dependency."
            " Install umbrela with `pyserini` support to use qrel-backed workflows."
        )
        if needs_java:
            message += " Java 21 is also required for Lucene-backed passage access."
        if _PYSERINI_IMPORT_ERROR is not None:
            message += f" Original import error: {_PYSERINI_IMPORT_ERROR}"
        raise ImportError(message) from _PYSERINI_IMPORT_ERROR


def get_catwise_data(
    qrel_data: QrelsData, few_shot_count: int
) -> list[tuple[QrelKey, QrelKey]]:
    examples: list[tuple[QrelKey, QrelKey]] = []
    for cat in [0, 1, 2, 3]:
        req_tuple_list: list[tuple[QrelKey, QrelKey]] = []

        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    req_tuple_list.append((qid, doc_id))
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")

        assert len(req_tuple_list) >= few_shot_count, (
            "Count of judgments available for category "
            f"{cat} is lesser than {few_shot_count}."
        )

        if len(req_tuple_list):
            samples_for_examples = random.sample(req_tuple_list, few_shot_count)
            examples.extend(samples_for_examples)
    return examples


def examples_prompt(
    few_shot_examples: list[tuple[QrelKey, QrelKey]],
    query_mappings: QueryMappings,
    qrel: str,
    qrel_data: QrelsData,
) -> str:
    prompt_examples = ""

    for example in few_shot_examples:
        query = query_mappings[int(example[0])]["title"]
        passage = get_passage_wrapper(qrel, example[1])

        res_json = f"##final score: {int(qrel_data[example[0]][example[1]])}"
        prompt_examples += f"""
                ###

                Query: {query}
                Passage: {passage}
                {res_json}
                """
    return prompt_examples


def get_query_mappings(qrel: str) -> QueryMappings:
    _require_pyserini("Built-in qrel topic lookup")

    # Query mappings
    topic_mapping = {
        "dl19-passage": "dl19-passage",
        "dl20-passage": "dl20-passage",
        "dl21-passage": "dl21",
        "dl22-passage": "dl22",
        "dl23-passage": "dl23",
        "robust04": "robust04",
        "robust05": "robust05",
    }
    if qrel not in topic_mapping:
        raise ValueError(f"Invalid value for qrel: {qrel}")
    query_mappings = cast(QueryMappings, get_topics(topic_mapping[qrel]))
    return query_mappings


def generate_examples_prompt(qrel: str, few_shot_count: int) -> str:
    qrel_data = get_qrels(qrel)
    few_shot_examples = get_catwise_data(qrel_data, few_shot_count)
    query_mappings = get_query_mappings(qrel)
    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, qrel, qrel_data
    )
    return prompt_examples


def generate_holes(
    qrel: str,
    judge_cat: list[int] | None = None,
    exception_qid: list[QrelKey] | None = None,
) -> tuple[list[tuple[QrelKey, QrelKey]], list[int]]:
    if judge_cat is None:
        judge_cat = [0, 1, 2, 3]
    if exception_qid is None:
        exception_qid = []

    qrel_data = get_qrels(qrel)
    holes: list[tuple[QrelKey, QrelKey]] = []
    gts: list[int] = []
    for cat in judge_cat:
        req_tuple_list: list[tuple[QrelKey, QrelKey]] = []

        total_count = 0
        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    total_count += 1
                    if qid not in exception_qid:
                        req_tuple_list.append((qid, doc_id))

        samples = req_tuple_list
        print(f"No. of judgments for category {cat}: {len(req_tuple_list)}")
        holes += samples
        gts += [cat] * len(samples)
    return holes, gts


def get_qrel_path(qrel_info: str) -> str:
    if not os.path.exists(qrel_info):
        _require_pyserini("Built-in qrel lookup")
        return cast(str, get_qrels_file(qrel_info))
    return qrel_info


def get_qrels(qrel_info: str) -> QrelsData:
    """This function is modified version of pyserini's get_qrels."""
    file_path = get_qrel_path(qrel_info)

    qrels: QrelsData = {}
    with open(file_path) as f:
        for line in f:
            qid, _, docid, judgement = line.rstrip().split()

            if qid.isdigit():
                qrels_key: QrelKey = int(qid)
            else:
                qrels_key = qid

            if docid.isdigit():
                doc_key: QrelKey = int(docid)
            else:
                doc_key = docid

            if qrels_key in qrels:
                qrels[qrels_key][doc_key] = judgement
            else:
                qrels[qrels_key] = {doc_key: judgement}
    return qrels


@lru_cache
def get_index_reader(qrel: str) -> Any | None:
    # Index reader
    if qrel in ["dl19-passage", "dl20-passage"]:
        _require_pyserini("MS MARCO v1 passage lookup", needs_java=True)
        index_reader = LuceneIndexReader.from_prebuilt_index("msmarco-v1-passage")
    else:
        index_reader = None
    return index_reader


def get_passage_msv2(pid: str) -> str:
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(
        f"../data/msmarco_v2_passage/msmarco_passage_{bundlenum}", encoding="utf8"
    ) as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid
        return cast(str, document["passage"])


def get_passage_wrapper(qrel: str, doc_id: QrelKey) -> str:
    index_reader = get_index_reader(qrel)
    if qrel in ["dl19-passage", "dl20-passage"]:
        if index_reader is None:
            raise RuntimeError("Lucene index reader was not initialized.")
        raw_document = json.loads(index_reader.doc_raw(str(doc_id)))
        passage = cast(str, raw_document.get("contents", ""))
    elif qrel in ["dl21", "dl22", "dl23"]:
        passage = get_passage_msv2(str(doc_id))
    else:
        raise ValueError(f"Invalid value for qrel: {qrel}")
    return passage


def prepare_query_passage(
    qid_docid_list: list[tuple[QrelKey, QrelKey]], qrel: str
) -> list[tuple[str, str]]:
    query_mappings = get_query_mappings(qrel)
    query_passage: list[tuple[str, str]] = []
    for sample in qid_docid_list:
        passage = get_passage_wrapper(qrel, sample[1])
        query_passage.append((query_mappings[int(sample[0])]["title"], passage))
    return query_passage


def fetch_ndcg_score(qrel_path: str, result_path: str) -> str | int:
    # -remove-unjudged
    _require_pyserini("nDCG evaluation")
    cmd = (
        "python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 "
        f"{qrel_path} {result_path}"
    ).split(" ")
    shell = platform.system() == "Windows"
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    stdout, stderr = process.communicate()
    output = stdout.decode("utf-8").rstrip()
    print(output)

    pattern = r"ndcg_cut_10\s+all\s+([0-9.]+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return 0


def get_dropped_cat_count(qrel: str, removal_fraction: float) -> dict[str, int]:
    qrel_data = get_qrels(qrel)

    cat_dict: dict[str, int] = {}
    for cat in [0, 1, 2, 3]:
        req_tuple_list: list[tuple[QrelKey, QrelKey]] = []

        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    req_tuple_list.append((qid, doc_id))

        print(
            f"No. of judgments for category {cat}: {len(req_tuple_list)}. "
            "Judgments that remain intact: "
            f"{len(req_tuple_list) - int(len(req_tuple_list) * removal_fraction)}"
        )
        cat_dict[str(cat)] = len(req_tuple_list) - int(
            len(req_tuple_list) * removal_fraction
        )
    return cat_dict
