from abc import ABC, abstractmethod
import pkg_resources
import os
import time

from umbrela.utils import qrel_utils



class LLMJudge(ABC):
    def __init__(
        self,
        qrel,
        prompt_file,
        prompt_type,
        model_name,
        few_shot_count,
    ) -> None:
        assert not (
            prompt_file and prompt_type
        ), "Both prompt_file and prompt_type passed. Only one mode must be selected!!"

        self.qrel = qrel

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

    def evalute_results_with_qrel(self, result_file, removal_fraction=0.9, removal_cat=[1, 2, 3]):
        holes = qrel_utils.generate_holes(self.qrel, removal_fraction, removal_cat)
        qrel_data = qrel_utils.get_qrels(self.qrel)

        valid_res_count = {}
        holes_qp = []
        gts = []
        holes_tup = []
        for cat in holes:
            holes_tup += holes[cat]
            holes_qp += qrel_utils.prepare_query_passage(holes[cat], self.qrel)
            gts += [cat] * len(holes[cat])
        judgments = self.judge(holes_qp, prepocess=False)

        for judgment, pair, gt in zip(judgments, holes_tup, gts):
            curr_res = int(gt == judgment["judgement"])
            if cat not in valid_res_count:
                valid_res_count[cat] = curr_res
            else:
                valid_res_count[cat] += curr_res
            qrel_data[pair[0]][pair[1]] = int(judgment["judgment"])
        
        for cat in valid_res_count:
            print(f"Stats for {cat}. Correct judgments count: {valid_res_count[cat]}/{len(holes[cat])}.")
        
        result_dir = f"modified_qrels/"
        os.makedirs(result_dir, exist_ok=True)

        path = qrel_utils.get_qrels_file(self.qrel)
        modified_qrel = f"{result_dir}/{os.path.basename(path)[:-4]}_{self.model_name}_{int(time.time())}"

        print(f"Output file: {modified_qrel}")
        
        with open(modified_qrel, "wb") as f_out:
            for qid in qrel_data:
                for doc_id in qrel_data[qid]:
                    result = str(qrel_data[qid][doc_id]) + "\n"
                    encoded = " ".join([str(qid), "0", doc_id, result]).encode("utf-8")
                    f_out.write(encoded)

        print("-"*79)
        output = {}
        output["original"] = qrel_utils.fetch_ndcf_score(self.qrel, result_file)
        output[f"modified_{int(removal_fraction * 100)}"] = qrel_utils.fetch_ndcf_score(self.qrel, result_file)
        print(output)
