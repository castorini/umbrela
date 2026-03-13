from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


JUDGE_COMMAND = CommandSpec(
    name="judge",
    description=(
        "Run an Umbrela backend on either direct JSON input "
        "or batch request files."
    ),
)
EVALUATE_COMMAND = CommandSpec(
    name="evaluate",
    description="Run qrel-backed evaluation and generate modified qrels plus metrics.",
)
DESCRIBE_COMMAND = CommandSpec(
    name="describe",
    description="Describe command contracts, examples, and defaults.",
)
SCHEMA_COMMAND = CommandSpec(
    name="schema",
    description="Print JSON schemas for inputs, outputs, and the shared CLI envelope.",
)
DOCTOR_COMMAND = CommandSpec(
    name="doctor",
    description="Report environment readiness for Umbrela backends and qrel workflows.",
)
VALIDATE_COMMAND = CommandSpec(
    name="validate",
    description=(
        "Validate command payloads and file requirements "
        "without executing a model."
    ),
)
