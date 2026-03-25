import asyncio
import os
import statistics
from abc import ABC, abstractmethod
from typing import Any, cast

from umbrela.prompts import (
    PromptTemplate,
    display_prompt_template,
    get_prompt_template,
    render_prompts,
)
from umbrela.utils import common_utils


class LLMJudge(ABC):
    def __init__(
        self,
        qrel: str,
        model_name: str,
        prompt_file: str | None,
        prompt_type: str | None,
        few_shot_count: int,
    ) -> None:
        assert not (prompt_file and prompt_type), (
            "Both prompt_file and prompt_type passed. Only one mode must be selected!!"
        )

        self.qrel = qrel
        self.few_shot_count = few_shot_count
        self.prompt_file = prompt_file
        self.model_name = model_name
        if few_shot_count > 0:
            from umbrela.utils import qrel_utils

            self.prompt_examples = qrel_utils.generate_examples_prompt(
                qrel, few_shot_count
            )
        elif few_shot_count == 0:
            self.prompt_examples = ""
            if prompt_file and "fewshot" in prompt_file:
                print(
                    "Warning!! default fewshot prompt file being used for "
                    "few_shot_count = 0"
                )
        else:
            raise ValueError(f"Invalid value for few_shot_count: {few_shot_count}")

        self.prompt_template: PromptTemplate = get_prompt_template(
            prompt_file=prompt_file,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
        )

    def display_prompt_template(self) -> None:
        display_prompt_template(self.prompt_template)

    def prepare_request_inputs(
        self, request_dict: dict[str, Any] | common_utils.QueryPassage, preprocess: bool
    ) -> tuple[common_utils.QueryPassage, list[str]]:
        query_passage = common_utils.prepare_request_inputs(
            request_dict,
            preprocess,
        )
        prompts = render_prompts(
            self.prompt_template, query_passage, self.prompt_examples
        )
        self.query_passage = query_passage
        self.prompts = prompts
        return query_passage, prompts

    def prepare_judgments(self, outputs: list[str]) -> list[common_utils.Judgment]:
        return common_utils.prepare_judgments(
            outputs,
            self.query_passage,
            self.prompts,
            self.model_name,
            getattr(self, "reasoning_outputs", None),
        )

    @abstractmethod
    def predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]: ...

    async def async_predict_with_llm(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int,
        preprocess: bool,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.predict_with_llm, request_dict, max_new_tokens, preprocess
        )

    @abstractmethod
    def judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]: ...

    async def async_judge(
        self,
        request_dict: dict[str, Any] | common_utils.QueryPassage,
        max_new_tokens: int = 100,
        preprocess: bool = True,
    ) -> list[common_utils.Judgment]:
        outputs = await self.async_predict_with_llm(
            request_dict, max_new_tokens, preprocess
        )
        return self.prepare_judgments(outputs)

    def evalute_results_with_qrel(
        self,
        result_file: str | None,
        judge_cat: list[int] | None = None,
        regenerate: bool = False,
        num_samples: int = 1,
        return_results_path: bool = False,
    ) -> str | None:
        from umbrela.utils import qrel_utils

        if judge_cat is None:
            judge_cat = [0, 1, 2, 3]

        result_dir = "modified_qrels"
        os.makedirs(result_dir, exist_ok=True)

        path = qrel_utils.get_qrels_file(self.qrel)
        modified_qrel = (
            f"{result_dir}/{os.path.basename(path)[:-4]}_"
            f"{self.model_name.split('/')[-1]}_{''.join(map(str, judge_cat))}_"
            f"{self.few_shot_count}_{num_samples}.txt"
        )
        print(f"Output file: {modified_qrel}")

        unmatch_dict: dict[int | str, list[int]] = {}
        if os.path.exists(modified_qrel) and not regenerate:
            org_qd = qrel_utils.get_qrels(self.qrel)
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
                self.qrel, judge_cat=judge_cat
            )
            qrel_data = cast(
                dict[int | str, dict[int | str, int | str]],
                qrel_utils.get_qrels(self.qrel),
            )
            holes_qp = qrel_utils.prepare_query_passage(holes_tup, self.qrel)
            if num_samples > 1:
                holes_qp = [item for item in holes_qp for _ in range(num_samples)]
                holes_tup = [item for item in holes_tup for _ in range(num_samples)]
                generated_gts = [
                    item for item in generated_gts for _ in range(num_samples)
                ]
            gts = list(generated_gts)

            judgments = self.judge(holes_qp, preprocess=False, max_new_tokens=200)

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
        common_utils.draw_confusion_matrix(gts, preds, self.qrel, self.model_name)

        for cat in unmatch_dict:
            print(
                "Stats for "
                f"{cat}. Correct judgments count: {sum(unmatch_dict[cat])}/"
                f"{len(unmatch_dict[cat])}"
            )

        if result_file:
            print("-" * 79)
            output = {}
            output["original"] = qrel_utils.fetch_ndcg_score(self.qrel, result_file)
            output["modified"] = qrel_utils.fetch_ndcg_score(modified_qrel, result_file)
            print(output)

        if return_results_path:
            return modified_qrel
        return None
