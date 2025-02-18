import argparse
from collections import Counter
import os

from dotenv import load_dotenv

from umbrela.gemini_judge import GeminiJudge
from umbrela.gpt_judge import GPTJudge
from umbrela.hgfllm_judge import HGFLLMJudge
from umbrela.osllm_judge import OSLLMJudge
from umbrela.utils import common_utils
from umbrela.utils import qrel_utils, common_utils

# Select relevance categories to be judged.
JUDGE_CAT = [0, 1, 2, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel", type=str, help="qrels file", required=True)
    parser.add_argument("--result_file", type=str, help="retriever result file")
    parser.add_argument("--prompt_file", type=str, help="prompt file")
    parser.add_argument(
        "--prompt_type", type=str, help="Prompt type. Supported types: [bing, basic]."
    )
    parser.add_argument("--llm_judges", type=str, help="LLM judges (, separated)")
    parser.add_argument("--model_names", type=str, help="model names (, separated)")
    parser.add_argument(
        "--few_shot_count", type=int, help="Few shot count for each category."
    )
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--regenerate", action="store_true")

    args = parser.parse_args()
    load_dotenv()

    result_dir = f"modified_qrels"
    os.makedirs(result_dir, exist_ok=True)

    llm_judges = args.llm_judges.split(",")
    model_names = args.model_names.split(",")

    assert len(llm_judges) == len(
        model_names
    ), "incomplete list of LLM judges or model names"

    results = []
    for i in range(len(llm_judges)):
        llm_judge = llm_judges[i]
        llm_judge = llm_judge.strip()
        llm_judges[i] = llm_judge
        try:
            cls = globals().get(llm_judge)
        except:
            raise ValueError(f"Invalid value for llm_judge: {llm_judge}")
        judge = cls(
            args.qrel,
            model_names[i].strip(),
            args.prompt_file,
            args.prompt_type,
            args.few_shot_count,
        )
        output_file = judge.evalute_results_with_qrel(
            args.result_file,
            regenerate=args.regenerate,
            num_samples=args.num_sample,
            return_results_path=True,
            judge_cat=JUDGE_CAT,
        )
        results.append(qrel_utils.get_qrels(output_file))

    final_qd = {}
    for qid in results[0]:
        for doc_id in results[0][qid]:
            if qid not in final_qd:
                final_qd[qid] = {}
            votes = [int(res[qid][doc_id]) for res in results]
            most_common = Counter(votes).most_common()
            max_count = most_common[0][1]
            best_id = min([id for id, count in most_common if count == max_count])
            final_qd[qid][doc_id] = best_id

    combined_model_name = "-".join(
        model_names[i].strip().split("/")[-1] for i in range(len(model_names))
    )
    path = qrel_utils.get_qrels_file(args.qrel)
    modified_qrel = f"{result_dir}/{os.path.basename(path)[:-4]}_{combined_model_name}_{''.join(map(str, JUDGE_CAT))}_{args.few_shot_count}_{args.num_sample}.txt"
    common_utils.write_modified_qrel(final_qd, modified_qrel)
    print("-" * 79)
    print("-" * 79)
    print(f"Output file: {modified_qrel}")
    print("-" * 79)

    org_qd = qrel_utils.get_qrels(args.qrel)
    unmatch_dict = {}
    gts, preds = [], []
    for qid in org_qd:
        for docid in org_qd[qid]:
            gts.append(int(org_qd[qid][docid]))
            preds.append(int(final_qd[qid][docid]))
            curr_res = int(int(org_qd[qid][docid]) == int(final_qd[qid][docid]))
            if int(org_qd[qid][docid]) not in unmatch_dict:
                unmatch_dict[int(org_qd[qid][docid])] = [curr_res]
            else:
                unmatch_dict[int(org_qd[qid][docid])].append(curr_res)

    common_utils.calculate_kappa(gts, preds)
    common_utils.draw_confusion_matrix(gts, preds, args.qrel, combined_model_name)
    for cat in unmatch_dict:
        print(
            f"Stats for {cat}. Correct judgments count: {sum(unmatch_dict[cat])}/{len(unmatch_dict[cat])}"
        )


if __name__ == "__main__":
    main()
