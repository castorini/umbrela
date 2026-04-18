from __future__ import annotations

import os
import statistics
from typing import Any, cast

from umbrela.utils import common_utils, qrel_utils


def evaluate_judge_results(
    judge: Any,
    *,
    result_file: str | None,
    judge_cat: list[int] | None = None,
    regenerate: bool = False,
    num_samples: int = 1,
    return_results_path: bool = False,
) -> str | None:
    if judge_cat is None:
        judge_cat = [0, 1, 2, 3]

    result_dir = "modified_qrels"
    os.makedirs(result_dir, exist_ok=True)

    path = qrel_utils.get_qrels_file(judge.qrel)
    modified_qrel = (
        f"{result_dir}/{os.path.basename(path)[:-4]}_"
        f"{judge.model_name.split('/')[-1]}_{''.join(map(str, judge_cat))}_"
        f"{judge.few_shot_count}_{num_samples}.txt"
    )
    print(f"Output file: {modified_qrel}")

    unmatch_dict: dict[int | str, list[int]] = {}
    if os.path.exists(modified_qrel) and not regenerate:
        org_qd = qrel_utils.get_qrels(judge.qrel)
        new_qd = qrel_utils.get_qrels(modified_qrel)

        gts: list[int | str] = []
        preds: list[int | str] = []

        for qid in org_qd:
            for docid in org_qd[qid]:
                if org_qd[qid][docid] not in unmatch_dict:
                    unmatch_dict[org_qd[qid][docid]] = []
                unmatch_dict[org_qd[qid][docid]].append(
                    int(org_qd[qid][docid] == new_qd[qid][docid])
                )
                gts.append(org_qd[qid][docid])
                preds.append(new_qd[qid][docid])

    else:
        holes_tup, generated_gts = qrel_utils.generate_holes(
            judge.qrel, judge_cat=judge_cat
        )
        qrel_data = cast(
            dict[int | str, dict[int | str, int | str]],
            qrel_utils.get_qrels(judge.qrel),
        )
        holes_qp = qrel_utils.prepare_query_passage(holes_tup, judge.qrel)
        if num_samples > 1:
            holes_qp = [item for item in holes_qp for _ in range(num_samples)]
            holes_tup = [item for item in holes_tup for _ in range(num_samples)]
            generated_gts = [item for item in generated_gts for _ in range(num_samples)]
        gts = list(generated_gts)

        judgments = judge.judge(holes_qp, preprocess=False, max_new_tokens=200)

        valid_res: dict[int | str, list[int]] = {}
        preds = []
        gts_valid: list[int | str] = []
        preds_valid: list[int | str] = []
        for index in range(0, len(judgments), num_samples):
            temp: list[int] = []
            for internal_index in range(index, index + num_samples):
                gt = gts[internal_index]
                judgment = judgments[internal_index]
                predicted_label = int(judgment["judgment"])
                preds.append(predicted_label)
                curr_res = int(gt == predicted_label)
                temp.append(predicted_label)
                if gt not in unmatch_dict:
                    unmatch_dict[gt] = [curr_res]
                else:
                    unmatch_dict[gt].append(curr_res)
                if judgment["result_status"]:
                    gts_valid.append(gt)
                    preds_valid.append(predicted_label)
                    if gt not in valid_res:
                        valid_res[gt] = [curr_res]
                    else:
                        valid_res[gt].append(curr_res)
            pair = holes_tup[index]
            qrel_data[pair[0]][pair[1]] = int(statistics.mode(temp))

        common_utils.write_modified_qrel(qrel_data, modified_qrel)
        print("For valid results:")
        common_utils.calculate_kappa(gts_valid, preds_valid)
        for cat in valid_res:
            print(
                "Stats for "
                f"{cat}. Correct judgments count in valid result: "
                f"{sum(valid_res[cat])}/{len(valid_res[cat])}"
            )

    print("For overall results:")
    common_utils.calculate_kappa(gts, preds)
    common_utils.draw_confusion_matrix(gts, preds, judge.qrel, judge.model_name)

    for cat in unmatch_dict:
        print(
            "Stats for "
            f"{cat}. Correct judgments count: {sum(unmatch_dict[cat])}/"
            f"{len(unmatch_dict[cat])}"
        )

    if result_file:
        print("-" * 79)
        output = {}
        output["original"] = qrel_utils.fetch_ndcg_score(judge.qrel, result_file)
        output["modified"] = qrel_utils.fetch_ndcg_score(modified_qrel, result_file)
        print(output)

    if return_results_path:
        return modified_qrel
    return None
