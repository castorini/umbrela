"""YAML-backed prompt loading and rendering for umbrela."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

REQUIRED_CUSTOM_PLACEHOLDERS = frozenset({"examples", "query", "passage"})
REQUIRED_BUILTIN_PLACEHOLDERS = frozenset({"query", "passage"})
SUPPORTED_PROMPT_TYPES = frozenset({"bing", "basic"})


@dataclass(frozen=True)
class PromptTemplate:
    """In-memory representation of a single prompt template."""

    method: str
    system_message: str
    prefix_user: str
    source_path: str

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for _, field_name, _, _ in Formatter().parse(self.prefix_user)
            if field_name is not None
        )

    def validate_required_placeholders(self, required: frozenset[str]) -> None:
        missing = required - set(self.placeholders)
        if missing:
            missing_display = ", ".join(f"`{{{name}}}`" for name in sorted(missing))
            raise ValueError(
                f"Prompt template must provide the fields {missing_display}."
            )

    def render(self, *, examples: str, query: str, passage: str) -> str:
        return self.prefix_user.format(
            examples=examples,
            query=query,
            passage=passage,
        )

    def raw_parts(self) -> dict[str, str]:
        return {
            "system_message": self.system_message,
            "prefix_user": self.prefix_user,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_path": self.source_path,
            "system_message": self.system_message,
            "prefix_user": self.prefix_user,
            "placeholders": list(self.placeholders),
        }


def _normalize_template_data(data: Any, source_path: Path) -> PromptTemplate:
    if not isinstance(data, dict):
        raise ValueError(f"Prompt template at {source_path} must be a YAML mapping.")

    prefix_user = data.get("prefix_user")
    if not isinstance(prefix_user, str):
        raise ValueError(
            f"Prompt template at {source_path} must define `prefix_user` as a string."
        )

    system_message = data.get("system_message", "")
    if not isinstance(system_message, str):
        raise ValueError(
            "Prompt template at "
            f"{source_path} must define `system_message` as a string."
        )

    method = data.get("method", "qrel")
    if not isinstance(method, str):
        raise ValueError(
            f"Prompt template at {source_path} must define `method` as a string."
        )

    return PromptTemplate(
        method=method,
        system_message=system_message,
        prefix_user=prefix_user,
        source_path=str(source_path),
    )


@lru_cache(maxsize=None)
def _load_template_from_path(template_path_str: str) -> PromptTemplate:
    template_path = Path(template_path_str)
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {template_path}")

    with open(template_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return _normalize_template_data(data, template_path)


def _resolve_builtin_template_path(prompt_type: str, few_shot_count: int) -> Path:
    if prompt_type not in SUPPORTED_PROMPT_TYPES:
        raise ValueError(f"Invalid prompt_type: {prompt_type}.")

    prompt_mode = "fewshot" if few_shot_count > 0 else "zeroshot"
    return Path(
        str(
            files("umbrela.prompts").joinpath(
                f"prompt_templates/qrel_{prompt_mode}_{prompt_type}.yaml"
            )
        )
    )


def get_prompt_template(
    prompt_file: str | None,
    prompt_type: str | None,
    few_shot_count: int,
) -> PromptTemplate:
    if prompt_file and prompt_type:
        raise AssertionError(
            "Both prompt_file and prompt_type passed. Only one mode must be selected!!"
        )

    if prompt_file is not None:
        template = _load_template_from_path(str(Path(prompt_file).resolve()))
        template.validate_required_placeholders(REQUIRED_CUSTOM_PLACEHOLDERS)
        return template

    if prompt_type is None:
        raise ValueError("A prompt file or supported prompt type is required.")

    template = _load_template_from_path(
        str(_resolve_builtin_template_path(prompt_type, few_shot_count))
    )
    template.validate_required_placeholders(REQUIRED_BUILTIN_PLACEHOLDERS)
    return template


def render_prompts(
    template: PromptTemplate,
    query_passage: list[tuple[str, str]],
    prompt_examples: str,
) -> list[str]:
    return [
        template.render(examples=prompt_examples, query=query, passage=passage)
        for query, passage in query_passage
    ]


def display_prompt_template(template: PromptTemplate) -> None:
    print(template.prefix_user)
