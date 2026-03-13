from __future__ import annotations

import asyncio
import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from umbrela.utils import common_utils, qrel_utils


@dataclass
class EvaluateResult:
    result_path: str
    stdout: str
    metrics: dict[str, Any]
    artifact_paths: list[str]


def create_judge(args: Any) -> Any:
    prompt_file = getattr(args, "prompt_file", None)
    prompt_type = getattr(args, "prompt_type", "bing")
    few_shot_count = getattr(args, "few_shot_count", 0)
    backend = getattr(args, "backend")
    model_name = getattr(args, "model")

    if backend == "gpt":
        from umbrela.gpt_judge import GPTJudge

        return GPTJudge(
            getattr(args, "qrel", "dl19-passage"),
            model_name,
            prompt_file,
            prompt_type,
            few_shot_count,
            use_azure_openai=getattr(args, "use_azure_openai", False),
            max_concurrency=getattr(args, "max_concurrency", 8),
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
    if backend == "gemini":
        from umbrela.gemini_judge import GeminiJudge

        return GeminiJudge(
            getattr(args, "qrel", "dl19-passage"),
            model_name,
            prompt_file,
            prompt_type,
            few_shot_count,
        )
    if backend == "hf":
        from umbrela.hgfllm_judge import HGFLLMJudge

        return HGFLLMJudge(
            getattr(args, "qrel", "dl19-passage"),
            model_name,
            prompt_file,
            prompt_type,
            few_shot_count,
            getattr(args, "device", "cuda"),
        )
    if backend == "os":
        from umbrela.osllm_judge import OSLLMJudge

        return OSLLMJudge(
            getattr(args, "qrel", "dl19-passage"),
            model_name,
            prompt_file,
            prompt_type,
            few_shot_count,
            getattr(args, "device", "cuda"),
        )
    if backend == "ensemble":
        raise ValueError("Ensemble execution is only supported through `evaluate`.")
    raise ValueError(f"Unsupported backend: {backend}")


def run_judge_direct(
    request_dict: dict[str, Any],
    args: Any,
) -> list[dict[str, Any]]:
    load_dotenv()
    judge = create_judge(args)
    execution_mode = getattr(args, "execution_mode", "sync")
    if execution_mode == "async":
        return cast(list[dict[str, Any]], asyncio.run(judge.async_judge(request_dict)))
    return cast(list[dict[str, Any]], judge.judge(request_dict))


def run_judge_batch(
    records: list[dict[str, Any]],
    args: Any,
) -> list[dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    for record in records:
        judgments.extend(run_judge_direct(record, args))
    return judgments


def _evaluate_with_single_judge(args: Any) -> EvaluateResult:
    load_dotenv()
    judge = create_judge(args)
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        result_path = judge.evalute_results_with_qrel(
            args.result_file,
            regenerate=args.regenerate,
            num_samples=args.num_sample,
            return_results_path=True,
            judge_cat=[int(value) for value in args.judge_cat.split(",")],
        )
    metrics: dict[str, Any] = {}
    if args.result_file and result_path:
        metrics = {
            "original_ndcg@10": qrel_utils.fetch_ndcg_score(
                args.qrel, args.result_file
            ),
            "modified_ndcg@10": qrel_utils.fetch_ndcg_score(
                result_path, args.result_file
            ),
        }
    artifact_paths = [path for path in [result_path] if path]
    conf_path = f"conf_matrix/{args.qrel}-{os.path.basename(args.model)}.png"
    if Path(conf_path).exists():
        artifact_paths.append(conf_path)
    return EvaluateResult(
        result_path=cast(str, result_path),
        stdout=output_capture.getvalue(),
        metrics=metrics,
        artifact_paths=artifact_paths,
    )


def run_evaluate(args: Any) -> EvaluateResult:
    if args.backend != "ensemble":
        return _evaluate_with_single_judge(args)

    load_dotenv()
    from collections import Counter

    llm_judges = [item.strip() for item in args.llm_judges.split(",")]
    model_names = [item.strip() for item in args.model_names.split(",")]
    if len(llm_judges) != len(model_names):
        raise ValueError("incomplete list of LLM judges or model names")

    results: list[dict[int | str, dict[int | str, str]]] = []
    artifact_paths: list[str] = []
    stdout_parts: list[str] = []
    for judge_backend, model_name in zip(llm_judges, model_names):
        nested_args = type("Args", (), vars(args).copy())()
        nested_args.backend = (
            judge_backend.removesuffix("Judge")
            .lower()
            .replace("hgfllm", "hf")
            .replace("osllm", "os")
        )
        nested_args.model = model_name
        result = _evaluate_with_single_judge(nested_args)
        stdout_parts.append(result.stdout)
        artifact_paths.extend(result.artifact_paths)
        results.append(qrel_utils.get_qrels(result.result_path))

    final_qd: dict[int | str, dict[int | str, int]] = {}
    for qid in results[0]:
        final_qd[qid] = {}
        for doc_id in results[0][qid]:
            votes = [int(result[qid][doc_id]) for result in results]
            most_common = Counter(votes).most_common()
            max_count = most_common[0][1]
            best_id = min(item for item, count in most_common if count == max_count)
            final_qd[qid][doc_id] = best_id

    result_dir = "modified_qrels"
    os.makedirs(result_dir, exist_ok=True)
    combined_model_name = "-".join(name.split("/")[-1] for name in model_names)
    path = qrel_utils.get_qrels_file(args.qrel)
    modified_qrel = (
        f"{result_dir}/{os.path.basename(path)[:-4]}_{combined_model_name}_"
        f"{args.judge_cat.replace(',', '')}_{args.few_shot_count}_{args.num_sample}.txt"
    )
    common_utils.write_modified_qrel(final_qd, modified_qrel)
    artifact_paths.append(modified_qrel)
    return EvaluateResult(
        result_path=modified_qrel,
        stdout="".join(stdout_parts),
        metrics={},
        artifact_paths=artifact_paths,
    )
