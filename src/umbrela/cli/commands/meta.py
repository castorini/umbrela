from __future__ import annotations

from pathlib import Path

from umbrela.cli.adapters import make_data_artifact
from umbrela.cli.introspection import COMMAND_DESCRIPTIONS, SCHEMAS, doctor_report
from umbrela.cli.responses import CommandResponse


def run_describe_command(target: str) -> CommandResponse:
    return CommandResponse(
        command="describe",
        mode="inspect",
        artifacts=[make_data_artifact(target, COMMAND_DESCRIPTIONS[target])],
    )


def run_schema_command(target: str) -> CommandResponse:
    return CommandResponse(
        command="schema",
        mode="inspect",
        artifacts=[make_data_artifact(target, SCHEMAS[target])],
    )


def run_doctor_command(*, config_path: Path | None = None) -> CommandResponse:
    report = doctor_report()
    report["config_file"] = str(config_path) if config_path else None
    return CommandResponse(
        command="doctor",
        mode="inspect",
        metrics=report,
        validation={"python_ok": report["python_ok"]},
    )
