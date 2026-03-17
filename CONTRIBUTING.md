# Contributing to umBRELA

Thank you for contributing to umBRELA. This repository packages large-language-model relevance judging, qrel-backed evaluation, and structured CLI workflows, so changes should preserve judgment semantics, artifact compatibility, and deterministic offline test coverage.

## Before You Start

- Open or reference a GitHub issue for significant bug fixes, features, or refactors when possible.
- Keep pull requests scoped to a single behavioral change or one tightly related maintenance update.
- If a change affects the `umbrela` CLI, prompt templates, evaluation outputs, or judge behavior, update the README, examples, and any relevant docs in the same pull request.

## Development Setup

umBRELA uses `uv` and the `dev` dependency group from `pyproject.toml`.

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
pre-commit install
```

Install backend extras only when you need them:

```bash
uv sync --group dev --extra cloud
uv sync --group dev --extra hf
uv sync --group dev --extra fastchat
uv sync --group dev --extra all
```

If you prefer not to activate the virtual environment, run commands through `uv run`.

## Local Quality Gate

Run these commands before opening a pull request:

```bash
uv run pre-commit run --all-files
uv sync --group dev --extra cloud
uv run pytest -q \
  tests/test_cli_main.py \
  tests/test_cli_evaluate.py \
  tests/test_cli_support.py \
  tests/test_evaluation_utils.py \
  tests/test_gpt_judge_async.py \
  tests/test_prompt_templates.py
uv run pytest -q tests/test_backend_contracts.py
```

The pre-commit hooks are the canonical lint, format, and type-check entrypoint for this repository. They currently run Ruff and MyPy.

## Testing Expectations

- Add or update tests for non-trivial behavior changes.
- Keep tests in one of these layers:
  - `core`: fast deterministic CLI, parsing, and envelope coverage that always runs in PR CI
  - `integration`: deterministic offline backend-contract regressions
  - `live`: provider-backed smoke tests gated behind explicit environment variables
- Prefer deterministic tests under `tests/` that do not require live model calls.
- If you change prompt rendering, output parsing, qrel rewriting, or CLI normalization, add focused regression coverage.
- Keep provider-specific or heavyweight validation as optional manual checks unless the scenario can be mocked cleanly.

## Prompt, CLI, and Evaluation Safety

- Do not silently change the structure of the `castorini.cli.v1` JSON envelope or the legacy judgment payload nested inside it.
- Preserve prompt semantics when editing YAML prompt templates under `src/umbrela/prompts/prompt_templates/`. If you intentionally change judge behavior, call it out explicitly in the pull request.
- Do not change qrel output naming, confusion-matrix artifacts, or score parsing heuristics without documenting downstream effects.
- Never hardcode provider credentials. Use `.env` or environment variables for OpenAI, Azure OpenAI, Vertex AI, Hugging Face, and FastChat-backed workflows.

## Documentation Expectations

- Update `README.md` for install, CLI, prompt-template, and environment changes.
- Update examples if the recommended invocation path changes.
- Document any new optional dependency requirements or backend assumptions in the pull request description.
- Add or update a file in `docs/release-notes/` for user-visible changes.
- If prompt semantics, parsing rules, backend defaults, or evaluation artifacts change, include a migration note in the release note.

## Pull Request Checklist

Before submitting:

1. Run the local quality gate commands listed above.
2. Summarize the user-visible behavior change and any prompt or evaluation impact.
3. Mention any artifact-path, schema, or dependency changes.
4. Include benchmarks or comparison data when the change may affect evaluation quality, latency, or cost.

## Reporting Issues

GitHub issues are the public tracker for bugs and feature requests. Good reports include:

- the exact `umbrela ...` command used
- the backend, model, and prompt type
- a minimal request payload, qrel, or run file
- expected versus observed behavior
- relevant logs, warnings, or output artifacts

## License

By contributing to umBRELA, you agree that your contributions will be licensed under the `LICENSE` file in the root of this repository.
