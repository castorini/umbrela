from __future__ import annotations

from pathlib import Path
from typing import Any

from umbrela.prompts import PromptTemplate, get_prompt_template


def _build_prompt_selector(
    *,
    prompt_file: str | None,
    prompt_type: str | None,
    few_shot_count: int,
    candidate_index: int | None = None,
    qrel: str | None = None,
) -> dict[str, Any]:
    selector: dict[str, Any] = {
        "prompt_file": str(Path(prompt_file).resolve()) if prompt_file else None,
        "prompt_type": prompt_type,
        "few_shot_count": few_shot_count,
    }
    if candidate_index is not None:
        selector["candidate_index"] = candidate_index
    if qrel is not None:
        selector["qrel"] = qrel
    return selector


def list_prompt_templates() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for prompt_type in ("basic", "bing"):
        for few_shot_count in (0, 1):
            template = get_prompt_template(
                prompt_file=None,
                prompt_type=prompt_type,
                few_shot_count=few_shot_count,
            )
            catalog.append(
                {
                    "selector": _build_prompt_selector(
                        prompt_file=None,
                        prompt_type=prompt_type,
                        few_shot_count=few_shot_count,
                    ),
                    "few_shot_mode": "fewshot" if few_shot_count > 0 else "zeroshot",
                    "source_path": template.source_path,
                    "method": template.method,
                    "placeholders": list(template.placeholders),
                }
            )
    return catalog


def build_prompt_template_view(
    template: PromptTemplate,
    *,
    prompt_file: str | None,
    prompt_type: str | None,
    few_shot_count: int,
) -> dict[str, Any]:
    return {
        "selector": _build_prompt_selector(
            prompt_file=prompt_file,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
        ),
        "template": template.metadata(),
    }


def render_prompt_catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = ["umbrela Prompt Catalog"]
    for entry in catalog:
        selector = entry["selector"]
        lines.append("")
        lines.append(
            f"- prompt_type={selector['prompt_type']} "
            f"few_shot_count={selector['few_shot_count']}"
        )
        lines.append(f"  mode: {entry['few_shot_mode']}")
        lines.append(f"  method: {entry['method']}")
        lines.append(f"  source: {entry['source_path']}")
        lines.append(f"  placeholders: {', '.join(entry['placeholders']) or '(none)'}")
    return "\n".join(lines)


def render_prompt_template_text(view: dict[str, Any]) -> str:
    selector = view["selector"]
    template = view["template"]
    lines = ["umbrela Prompt Template"]
    lines.append(f"method: {template['method']}")
    lines.append(f"source: {template['source_path']}")
    if selector["prompt_file"] is not None:
        lines.append(f"prompt_file: {selector['prompt_file']}")
    else:
        lines.append(f"prompt_type: {selector['prompt_type']}")
        lines.append(f"few_shot_count: {selector['few_shot_count']}")
    lines.append(
        "placeholders: "
        + (
            ", ".join(template["placeholders"])
            if template["placeholders"]
            else "(none)"
        )
    )
    lines.append("")
    lines.append("[system]")
    lines.append(template["system_message"] or "(empty)")
    lines.append("")
    lines.append("[user]")
    lines.append(template["prefix_user"])
    return "\n".join(lines)


def build_rendered_prompt_view(
    template: PromptTemplate,
    *,
    prompt_file: str | None,
    prompt_type: str | None,
    few_shot_count: int,
    candidate_index: int,
    qrel: str | None,
    query: str,
    passage: str,
    examples: str,
) -> dict[str, Any]:
    return {
        "selector": _build_prompt_selector(
            prompt_file=prompt_file,
            prompt_type=prompt_type,
            few_shot_count=few_shot_count,
            candidate_index=candidate_index,
            qrel=qrel,
        ),
        "messages": {
            "system": template.system_message,
            "user": template.render(
                examples=examples,
                query=query,
                passage=passage,
            ),
        },
        "inputs": {
            "query": query,
            "passage": passage,
            "examples": examples,
        },
    }


def render_rendered_prompt_text(view: dict[str, Any], *, part: str) -> str:
    selector = view["selector"]
    messages = view["messages"]
    inputs = view["inputs"]
    lines = ["umbrela Rendered Prompt"]
    if selector["prompt_file"] is not None:
        lines.append(f"prompt_file: {selector['prompt_file']}")
    else:
        lines.append(f"prompt_type: {selector['prompt_type']}")
        lines.append(f"few_shot_count: {selector['few_shot_count']}")
    if selector.get("qrel") is not None:
        lines.append(f"qrel: {selector['qrel']}")
    lines.append(f"candidate_index: {selector['candidate_index']}")
    lines.append(f"query: {inputs['query']}")
    lines.append(f"passage: {inputs['passage']}")
    if part in {"system", "all"}:
        lines.append("")
        lines.append("[system]")
        lines.append(messages["system"] or "(empty)")
    if part in {"user", "all"}:
        lines.append("")
        lines.append("[user]")
        lines.append(messages["user"])
    return "\n".join(lines)
