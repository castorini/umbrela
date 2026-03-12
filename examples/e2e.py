#!/usr/bin/env python3
"""Quick end-to-end example for judging a small query-passage set."""

import argparse
from textwrap import fill

from dotenv import load_dotenv


def create_sample_request() -> dict:
    """Create a small in-memory request that matches Umbrela's judge API."""
    return {
        "query": {"text": "how long is life cycle of flea", "qid": "264014"},
        "candidates": [
            {
                "doc": {
                    "segment": (
                        "The life cycle of a flea can last anywhere from 20 days to "
                        "an entire year. It depends on how long the flea remains in "
                        "the dormant stage (eggs, larvae, pupa)."
                    )
                },
                "docid": "4834547",
                "score": 14.971799850463867,
            },
            {
                "doc": {
                    "segment": (
                        "The flea egg stage is the beginning of the flea cycle. "
                        "Depending on temperature and humidity, eggs can take from "
                        "two to six days to hatch."
                    )
                },
                "docid": "6641238",
                "score": 15.090800285339355,
            },
            {
                "doc": {
                    "segment": (
                        "Flea larvae spin cocoons around themselves before becoming "
                        "adult fleas. The larvae can remain in the cocoon anywhere "
                        "from one week to one year."
                    )
                },
                "docid": "96852",
                "score": 14.215100288391113,
            },
            {
                "doc": {
                    "segment": (
                        "A flea can live up to a year, but its general lifespan "
                        "depends on living conditions such as temperature and host "
                        "availability."
                    )
                },
                "docid": "5611210",
                "score": 15.780599594116211,
            },
        ],
    }


def build_judge(args):
    """Construct the requested judge backend with only the needed imports."""
    prompt_type = None if args.prompt_file else args.prompt_type

    if args.judge == "gpt":
        from umbrela.gpt_judge import GPTJudge

        return GPTJudge(
            qrel=args.qrel,
            model_name=args.model,
            prompt_file=args.prompt_file,
            prompt_type=prompt_type,
            few_shot_count=args.few_shot_count,
            use_azure_openai=args.use_azure_openai,
        )

    if args.judge == "gemini":
        from umbrela.gemini_judge import GeminiJudge

        return GeminiJudge(
            qrel=args.qrel,
            model_name=args.model,
            prompt_file=args.prompt_file,
            prompt_type=prompt_type,
            few_shot_count=args.few_shot_count,
        )

    if args.judge == "hf":
        from umbrela.hgfllm_judge import HGFLLMJudge

        return HGFLLMJudge(
            qrel=args.qrel,
            model_name=args.model,
            prompt_file=args.prompt_file,
            prompt_type=prompt_type,
            few_shot_count=args.few_shot_count,
            device=args.device,
            num_gpus=args.num_gpus,
        )

    from umbrela.osllm_judge import OSLLMJudge

    return OSLLMJudge(
        qrel=args.qrel,
        model_name=args.model,
        prompt_file=args.prompt_file,
        prompt_type=prompt_type,
        few_shot_count=args.few_shot_count,
        device=args.device,
        num_gpus=args.num_gpus,
    )


def print_results(
    request: dict,
    judgments: list[dict],
    show_raw: bool = False,
    passage_width: int = 100,
) -> None:
    """Render readable per-document results for manual smoke tests."""
    query = request["query"]["text"]
    qid = request["query"].get("qid", "unknown")

    print("Query")
    print(f"  qid: {qid}")
    print(f"  text: {query}")
    print()
    print(f"Received {len(judgments)} judgments:")
    for rank, (candidate, judgment) in enumerate(
        zip(request["candidates"], judgments), start=1
    ):
        label = judgment["judgment"]
        parsed = "yes" if judgment["result_status"] else "no"
        docid = candidate.get("docid", "unknown")
        score = candidate.get("score")
        passage = fill(judgment["passage"], width=passage_width)

        print(f"{rank}. label={label} parsed={parsed}")
        print(f"   docid: {docid}")
        if score is not None:
            print(f"   score: {score:.4f}")
        print("   passage:")
        for line in passage.splitlines():
            print(f"     {line}")
        if show_raw:
            print(f"   raw={judgment['prediction']!r}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small Umbrela end-to-end example against one judge."
    )
    parser.add_argument(
        "--judge",
        choices=["gpt", "gemini", "hf", "os"],
        default="gpt",
        help="Judge backend to run.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name for the selected backend.",
    )
    parser.add_argument(
        "--qrel",
        default="dl19-passage",
        help="Qrel identifier used to pick built-in few-shot examples.",
    )
    parser.add_argument(
        "--prompt_type",
        default="bing",
        choices=["bing", "basic"],
        help="Built-in prompt template to use when --prompt_file is not set.",
    )
    parser.add_argument(
        "--prompt_file",
        help="Optional custom prompt template path. Overrides --prompt_type.",
    )
    parser.add_argument(
        "--few_shot_count",
        type=int,
        default=0,
        help="Few-shot examples per label category. Use 0 for zero-shot.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum new tokens to request from the model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for local HF/FastChat backends.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="GPU count for local HF/FastChat backends.",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the first generated prompt after the run completes.",
    )
    parser.add_argument(
        "--print_raw",
        action="store_true",
        help="Print raw model responses for each passage.",
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="Use Azure OpenAI instead of the default public OpenAI API.",
    )
    parser.add_argument(
        "--passage_width",
        type=int,
        default=100,
        help="Wrap width for displayed passages.",
    )
    args = parser.parse_args()

    load_dotenv()
    request = create_sample_request()

    print(
        f"Running Umbrela e2e example with judge={args.judge} "
        f"model={args.model} prompt_type={args.prompt_type} "
        f"few_shot_count={args.few_shot_count}"
    )
    judge = build_judge(args)
    judgments = judge.judge(request, max_new_tokens=args.max_new_tokens)

    if args.print_prompt and judgments:
        print("\nFirst prompt:\n")
        print(judgments[0]["prompt"])

    print()
    print_results(
        request,
        judgments,
        show_raw=args.print_raw,
        passage_width=args.passage_width,
    )


if __name__ == "__main__":
    main()
