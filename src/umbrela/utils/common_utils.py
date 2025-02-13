import re
import os

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay


def preprocess_request_dict(request_dict):
    query_passage = []
    query = request_dict["query"]["text"]
    for cand in request_dict["candidates"]:
        query_passage.append((query, cand["doc"]["segment"]))
    return query_passage


def generate_prompts(query_passage, prompt_examples, prompt_template):
    prompts = []
    for q_p in query_passage:
        prompt = prompt_template.format(
            examples=prompt_examples,
            query=q_p[0],
            passage=q_p[1],
        )
        prompts.append(prompt)
    return prompts


def parse_fewshot_response(response: str, passage: str, query: str) -> int:
    response = response.strip().lower()
    valid_res = 1
    answer = ""
    patterns = [
        r'"o"\s*[:-=]?\s*(0|1|2|3)',
        r"\'o\'\s*[:-=]?\s*(0|1|2|3)",
        r"o\s*[:-=]?\s*(0|1|2|3)",
        r'"overall_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall"\s*[:-=]?\s*(0|1|2|3)',
        r'"overall score"\s*[:-=]?\s*(0|1|2|3)',
        r'"final score"\s*[:-=]?\s*(0|1|2|3)',
        r"final score\s*[:-=]?\s*(0|1|2|3)",
        r"final score is (0|1|2|3)",
        r'"final_score"\s*[:-=]?\s*(0|1|2|3)',
        r'"score"\s*[:-=]?\s*(0|1|2|3)',
        r'"o_score"\s*[:-=]?\s*(0|1|2|3)',
        r"output score is (0|1|2|3)",
        r"score is (0|1|2|3)",
        r"[a-zA-Z]+\s+is\s+(0|1|2|3)\s",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category\s*[:-=]?\s*(0|1|2|3)",
        r"relevance category is (0|1|2|3)",
        r"it falls into the category (0|1|2|3)",
        r"category\s*(0|1|2|3)",
        r"relevance category (0|1|2|3)",
        r"relevance category for this passage would be (0|1|2|3)",
        r"the relevance category would be (0|1|2|3)",
        r"\n*(0|1|2|3)",
    ]
    for pattern in patterns:
        matched = None
        for m in re.finditer(
            pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL
        ):
            matched = m

        if matched:
            answer = matched.group(1).capitalize()
            break
    if answer == "":
        answer = "0"
        valid_res = 0
        print(f"Invalid response to `{query}` & `{passage}`: {response}")
    return int(answer), valid_res


def prepare_judgments(outputs, query_passage, prompts, model_name):
    judgments = []
    for output, (query, passage), prompt in zip(outputs, query_passage, prompts):
        res = parse_fewshot_response(output, query, passage)
        judgment = {
            "model": model_name,
            "query": query,
            "passage": passage,
            "prompt": prompt,
            "prediction": output,
            "judgment": res[0],
            "result_status": res[1],
        }
        judgments.append(judgment)
    return judgments


def write_modified_qrel(modified_data, qrel_path):
    with open(qrel_path, "wb") as f_out:
        for qid in modified_data:
            for doc_id in modified_data[qid]:
                result = str(modified_data[qid][doc_id]) + "\n"
                encoded = " ".join([str(qid), "Q0", str(doc_id), str(result)]).encode(
                    "utf-8"
                )
                f_out.write(encoded)


def calculate_kappa(gts, preds):
    print(f"Kohen kappa overall: {cohen_kappa_score(gts, preds)}")
    print("-" * 79)
    gts_bin = [1 if int(x) > 1 else 0 for x in gts]
    preds_bin = [1 if int(x) > 1 else 0 for x in preds]
    print(f"Binarized Kohen kappa overall: {cohen_kappa_score(gts_bin, preds_bin)}")
    print("-" * 79)


def draw_confusion_matrix(gts, preds, qrel, model_name):
    conf_mat = confusion_matrix(gts, preds)
    print(conf_mat)

    os.makedirs("conf_matrix", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="GnBu")
    for text in disp.text_.ravel():
        text.set_fontsize(16)
    ax.set_title(qrel, fontsize=14)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    plt.savefig(f"conf_matrix/{qrel}-{os.path.basename(model_name)}.png")
