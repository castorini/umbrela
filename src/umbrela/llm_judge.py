from abc import ABC, abstractmethod
import pkg_resources
import os
import statistics
import time

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from umbrela.utils import qrel_utils, common_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str,
        prompt_type: str,
        few_shot_count: int,
    ) -> None:
        assert not (
            prompt_file and prompt_type
        ), "Both prompt_file and prompt_type passed. Only one mode must be selected!!"

        self.qrel = qrel
        self.few_shot_count = few_shot_count

        if prompt_type:
            if prompt_type not in ["bing", "basic"]:
                raise ValueError(f"Invalid prompt_type: {prompt_type}.")
            prompt_mode_str = "fewshot" if few_shot_count > 0 else "zeroshot"
            prompt_file = pkg_resources.resource_filename(
                "umbrela", f"prompts/qrel_{prompt_mode_str}_{prompt_type}.txt"
            )
            if not os.path.exists(prompt_file):
                raise ValueError(f"Prompt file doesn't exist.")

        if prompt_file:
            print(
                "Warning!! Prompt file expects input fields namely: (examples, query, passage)."
            )
        self.model_name = model_name
        if few_shot_count > 0:
            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel, few_shot_count
            )
        elif few_shot_count == 0:
            self.prompt_examples = ""
            if "fewshot" in prompt_file:
                print(
                    f"Warning!! default fewshot prompt file being used for few_shot_count = 0"
                )
        else:
            raise ValueError(f"Invalid value for few_shot_count: {few_shot_count}")

        with open(prompt_file) as p:
            self._prompt_template = "".join(p.readlines()).strip()

    def display_prompt_template(self):
        print(self._prompt_template)

    @abstractmethod
    def predict_with_llm(self, request_dict, max_new_tokens, prepocess):
        pass

    @abstractmethod
    def judge(self, request_dict, max_new_tokens=100, prepocess: bool = True):
        pass

    def calculate_kappa(self, gts, preds):
        print(f"Kohen kappa overall: {cohen_kappa_score(gts, preds)}")
        print("-" * 79)
        gts_bin = [1 if int(x) > 1 else 0 for x in gts]
        preds_bin = [1 if int(x) > 1 else 0 for x in preds]
        print(f"Binarized Kohen kappa overall: {cohen_kappa_score(gts_bin, preds_bin)}")
        print("-" * 79)

    def draw_confusion_matrix(self, gts, preds):
        conf_mat = confusion_matrix(gts, preds)
        print(conf_mat)

        os.makedirs("conf_matrix", exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap="GnBu")
        for text in disp.text_.ravel():
            text.set_fontsize(16)
        ax.set_title(self.qrel, fontsize=14)
        ax.set_xlabel("Predicted label", fontsize=14)
        ax.set_ylabel("True label", fontsize=14)
        plt.savefig(f"conf_matrix/{self.qrel}.png")

    def evalute_results_with_qrel(
        self,
        result_file,
        removal_cat=[0, 1, 2, 3],
        regenerate=False,
        num_samples=1,
    ):
        result_dir = f"modified_qrels"
        os.makedirs(result_dir, exist_ok=True)

        path = qrel_utils.get_qrels_file(self.qrel)
        modified_qrel = f"{result_dir}/{os.path.basename(path)[:-4]}_{self.model_name.split('/')[-1]}_{self.few_shot_count}_{num_samples}.txt"
        print(f"Output file: {modified_qrel}")

        if os.path.exists(modified_qrel) and not regenerate:
            org_qd = qrel_utils.get_qrels(self.qrel)
            new_qd = qrel_utils.get_qrels(modified_qrel)

            unmatch_dict = {}
            gts, preds = [], []

            for qid in org_qd:
                for docid in org_qd[qid]:
                    if org_qd[qid][docid] not in unmatch_dict:
                        unmatch_dict[org_qd[qid][docid]] = []
                    unmatch_dict[org_qd[qid][docid]].append(int(org_qd[qid][docid] == new_qd[qid][docid]))
                    gts.append(org_qd[qid][docid])
                    preds.append(new_qd[qid][docid])

        else:
            holes_tup, gts = qrel_utils.generate_holes(self.qrel, removal_cat=removal_cat)
            qrel_data = qrel_utils.get_qrels(self.qrel)
            unmatch_dict = {}
            holes_qp = qrel_utils.prepare_query_passage(holes_tup, self.qrel)
            if num_samples > 1:
                holes_qp = [item for item in holes_qp for _ in range(num_samples)]
                holes_tup = [item for item in holes_tup for _ in range(num_samples)]
                gts = [item for item in gts for _ in range(num_samples)]

            judgments = self.judge(holes_qp, prepocess=False, max_new_tokens=200)

            valid_res = {}
            preds = []
            gts_valid, preds_valid = [], []
            for index in range(0, len(judgments), num_samples):
                temp = []
                for internal_index in range(index, index + num_samples):
                    gt = gts[internal_index]
                    judgment = judgments[internal_index]
                    preds.append(judgment["judgment"])
                    curr_res = int(gt == judgment["judgment"])
                    temp.append(judgment["judgment"])
                    if gt not in unmatch_dict:
                        unmatch_dict[gt] = [curr_res]
                    else:
                        unmatch_dict[gt].append(curr_res)
                    if judgment["result_status"]:
                        gts_valid.append(gt)
                        preds_valid.append(judgment["judgment"])
                        if gt not in valid_res:
                            valid_res[gt] = [curr_res]
                        else:
                            valid_res[gt].append(curr_res)
                pair = holes_tup[index]
                qrel_data[pair[0]][pair[1]] = int(statistics.mode(temp))

            common_utils.write_modified_qrel(qrel_data, modified_qrel)
            print("For valid results:")
            self.calculate_kappa(gts_valid, preds_valid)
            for cat in valid_res:
                print(
                    f"Stats for {cat}. Correct judgments count in valid result: {sum(valid_res[cat])}/{len(valid_res[cat])}"
                )

        print("For overall results:")
        self.calculate_kappa(gts, preds)
        self.draw_confusion_matrix(gts, preds)

        for cat in unmatch_dict:
            print(
                f"Stats for {cat}. Correct judgments count: {sum(unmatch_dict[cat])}/{len(unmatch_dict[cat])}"
            )

        if result_file:
            print("-" * 79)
            output = {}
            output["original"] = qrel_utils.fetch_ndcf_score(self.qrel, result_file)
            output[f"modified"] = qrel_utils.fetch_ndcf_score(modified_qrel, result_file)
            print(output)
