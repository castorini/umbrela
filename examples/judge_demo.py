#!/usr/bin/env python3
"""Async-first GPT judge demo for umbrela."""

import argparse
import asyncio
from textwrap import fill

from dotenv import load_dotenv

from umbrela.gpt_judge import GPTJudge


def create_sample_request() -> dict:
    """Create a small in-memory request that matches umbrela's judge API."""
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


def uses_reasoning_style_model(model_name: str) -> bool:
    return (
        "o1" in model_name
        or "o3" in model_name
        or "o4" in model_name
        or "gpt-5" in model_name
    )


def print_results(
    request: dict,
    judgments: list[dict],
    show_raw: bool = False,
    show_reasoning: bool = False,
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
        if show_reasoning and judgment.get("reasoning"):
            print("   reasoning:")
            for line in fill(judgment["reasoning"], width=passage_width).splitlines():
                print(f"     {line}")
        if show_raw:
            print(f"   raw={judgment['prediction']!r}")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the default async GPT judge demo for umbrela."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI, OpenRouter, or Azure OpenAI model name to use.",
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
        help="Optional custom YAML prompt template path. Overrides --prompt_type.",
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
        "--max_concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent OpenAI requests.",
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
        "--print_reasoning",
        action="store_true",
        help="Print model reasoning content when the provider returns it.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help=(
            "Reasoning effort for OpenAI reasoning models such as gpt-5 "
            "and o-series models."
        ),
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="Use Azure OpenAI instead of the default public OpenAI API.",
    )
    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Use OpenRouter instead of the default public OpenAI API.",
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
    prompt_type = None if args.prompt_file else args.prompt_type
    reasoning_effort = args.reasoning_effort
    if (
        args.print_reasoning
        and reasoning_effort is None
        and uses_reasoning_style_model(args.model)
    ):
        reasoning_effort = "medium"
        print(
            "No reasoning effort specified; using reasoning_effort=medium "
            "for this reasoning-capable model."
        )

    print(
        f"Running umbrela async judge demo with model={args.model} "
        f"prompt_type={args.prompt_type} few_shot_count={args.few_shot_count} "
        f"max_concurrency={args.max_concurrency}"
    )
    judge = GPTJudge(
        qrel=args.qrel,
        model_name=args.model,
        prompt_file=args.prompt_file,
        prompt_type=prompt_type,
        few_shot_count=args.few_shot_count,
        use_azure_openai=args.use_azure_openai,
        use_openrouter=args.use_openrouter,
        max_concurrency=args.max_concurrency,
        reasoning_effort=reasoning_effort,
    )
    judgments = await judge.async_judge(request, max_new_tokens=args.max_new_tokens)

    if args.print_prompt and judgments:
        print("\nFirst prompt:\n")
        print(judgments[0]["prompt"])

    print()
    print_results(
        request,
        judgments,
        show_raw=args.print_raw,
        show_reasoning=args.print_reasoning,
        passage_width=args.passage_width,
    )


if __name__ == "__main__":
    asyncio.run(main())
