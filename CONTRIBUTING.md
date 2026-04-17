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
```

The repository pins Python 3.11 in `.python-version`, so `uv` will select that interpreter automatically when possible. `uv sync --group dev` installs the base project plus the default `dev` dependency group; optional extras remain opt-in even though they are represented in `uv.lock`.

For the full quality gate, also sync the cloud extra so MyPy can resolve the optional backend imports used by the gate:

```bash
uv sync --group dev --extra cloud
```

Install backend extras only when you need them:

```bash
uv sync --group dev --extra cloud
uv sync --group dev --extra hf
uv sync --group dev --extra fastchat
uv sync --group dev --extra all
```

If you prefer not to activate the virtual environment, prefer running commands through `uv run`.
Install the pre-commit and pre-push hooks with `pre-commit install --hook-type pre-commit --hook-type pre-push` if you want Git to invoke the local quality gate automatically.

## Local Quality Gate

Run these commands before opening a pull request:

```bash
bash scripts/quality_gate.sh commit
bash scripts/quality_gate.sh push
```

The repo-local quality gate script is the canonical lint, format, test, and type-check entrypoint for this repository. Both gate modes start with `uv lock --check`, then run Ruff, core tests, integration tests, and MyPy in that order. The installed Git hooks both run the non-mutating push-mode gate so commits and pushes see the same validation order. Use `bash scripts/quality_gate.sh commit` manually if you want Ruff autofixes before re-running the checks.

## Testing Expectations

- Add or update tests for non-trivial behavior changes.
- Keep tests in one of these layers:
  - `core`: fast deterministic CLI, parsing, and envelope coverage that always runs in PR CI
  - `integration`: deterministic offline backend-contract regressions
  - `live`: provider-backed smoke tests gated behind explicit environment variables
- Apply the shared pytest markers (`core`, `integration`, `live`) at the module level so CI and local commands stay aligned across Castorini Python repos.
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
