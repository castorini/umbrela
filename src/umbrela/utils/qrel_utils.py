from functools import lru_cache
import json
import os
import random
import re
import platform
import subprocess

#from pyserini.index.lucene import IndexReader
#from pyserini.search import get_qrels_file, get_topics


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


def examples_prompt(few_shot_examples, query_mappings, qrel, qrel_data):
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


def get_query_mappings(qrel):
    # Query mappings
    topic_mapping = {
        "dl19-passage": "dl19-passage",
        "dl20-passage": "dl20",
        "dl21-passage": "dl21",
        "dl22-passage": "dl22",
        "dl23-passage": "dl23",
        "robust04": "robust04",
        "robust05": "robust05"
    }
    if qrel not in topic_mapping:
        raise ValueError(f"Invalid value for qrel: {qrel}")
    query_mappings = get_topics(topic_mapping[qrel])
    return query_mappings

@lru_cache
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
    prompt_examples = examples_prompt(
        few_shot_examples, query_mappings, qrel, qrel_data
    )
    return prompt_examples

def generate_holes(qrel, removal_fraction, removal_cat, exception_qid = []):
    qrel_data = get_qrels(qrel)
    holes = []
    gts = []
    for cat in removal_cat:
        req_tuple_list = []

        total_count = 0
        for qid in qrel_data:
            for doc_id in qrel_data[qid]:
                if int(qrel_data[qid][doc_id]) == cat:
                    total_count += 1
                    if qid not in exception_qid:
                        req_tuple_list.append((qid, doc_id))
    
        sample_size = int(total_count * removal_fraction)
        samples = random.sample(req_tuple_list, sample_size)
        print(f"No. of holes created for category {cat}: {sample_size}")
        holes += samples
        gts += [cat] * len(samples)
    return holes, gts

def get_qrel_path(qrel_info):
    if not os.path.exists(qrel_info):
        return get_qrels_file(qrel_info)
    return qrel_info

def get_qrels(qrel_info):
    """This function is modified version of pyserini's get_qrels."""
    file_path = get_qrel_path(qrel_info)
        
    qrels = {}
    with open(file_path, 'r') as f:
        for line in f:
            qid, _, docid, judgement = line.rstrip().split()
            
            if qid.isdigit():
                qrels_key = int(qid)
            else:
                qrels_key = qid
                
            if docid.isdigit():
                doc_key = int(docid)
            else:
                doc_key = docid
                
            if qrels_key in qrels:
                qrels[qrels_key][doc_key] = judgement
            else:
                qrels[qrels_key] = {doc_key: judgement}
    return qrels

def get_passage_wrapper(qrel, doc_id):
    index_reader = get_index_reader(qrel)
    if qrel in ["dl19-passage", "dl20-passage"]:
        passage = json.loads(index_reader.doc_raw(str(doc_id))).get(
            "contents", ""
        )
    else:
        passage = get_passage(doc_id)
    return passage

def prepare_query_passage(qid_docid_list, qrel):
    query_mappings = get_query_mappings(qrel)
    query_passage = []
    for sample in qid_docid_list:
        passage = get_passage_wrapper(qrel, sample[1])
        query_passage.append((query_mappings[int(sample[0])]["title"], passage))
    return query_passage

def fetch_ndcf_score(qrel_path, result_path):
    # -remove-unjudged
    cmd = f"python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 {qrel_path} {result_path}"
    cmd = cmd.split(" ")
    shell = platform.system() == "Windows"
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=shell)
    stdout, stderr = process.communicate()
    output = stdout.decode("utf-8").rstrip()
    print(output)

    pattern = r'ndcg_cut_10\s+all\s+([0-9.]+)'
    match = re.search(pattern, output)
    if match:
        # Extract the value from the matched group
        return match.group(1)
    else:
        return 0


def get_dropped_cat_count(qrel, removal_fraction):
  qrel_data = get_qrels(qrel)
  
  cat_dict = {}
  for index, cat in enumerate([0, 1, 2, 3]):
      req_tuple_list = []

      total_count = 0
      for qid in qrel_data:
          for doc_id in qrel_data[qid]:
              if int(qrel_data[qid][doc_id]) == cat:
                  total_count += 1
                  req_tuple_list.append((qid, doc_id))

      print(
          f"No. of judgments for category {cat}: {len(req_tuple_list)}. Judgments that remain intact: {len(req_tuple_list) - int(len(req_tuple_list) * 0.9)}"
      )
      cat_dict[str(cat)] = len(req_tuple_list) - int(len(req_tuple_list) * removal_fraction)
  return cat_dict
