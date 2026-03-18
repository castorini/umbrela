---
name: umbrela-install
description: Set up an umbrela development environment — checks Python 3.11+, installs via uv or pip with cloud extras, and verifies with doctor. Use when someone is onboarding, setting up a fresh clone, or troubleshooting their environment.
---

# umbrela Install

Development environment setup for [umbrela](https://github.com/castorini/umbrela) — LLM-based relevance assessment of query-passage pairs.

## Prerequisites

- Python 3.11+
- Git (SSH access to `github.com:castorini`)

## Verify Runtime

```bash
python3 --version   # must be 3.11+
command -v uv       # if present, use uv path; otherwise recommend uv
```

If `uv` is on PATH, use it silently. If not, ask the user once: install uv or proceed with pip.

## Clone (if needed)

If no `pyproject.toml` in cwd:

```bash
git clone git@github.com:castorini/umbrela.git && cd umbrela
```

## Install (source — preferred)

### uv path

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev --extra cloud
```

### pip path

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cloud]"
pip install pre-commit pytest mypy ruff
```

### PyPI alternative (mention but don't default to)

```bash
pip install umbrela
```

## Smoke Test

```bash
umbrela doctor --output json
umbrela --help
```

## Pre-commit (source installs)

```bash
pre-commit install
```

## Reference Files

- `references/extras.md` — Optional dependency stacks (cloud, hf, fastchat, pyserini)

## Gotchas

- Java 21 is only needed if installing `--extra pyserini` for evaluation workflows. The core umbrela package does not require Java.
- Dev dependencies use PEP 735 `[dependency-groups]` — only `uv sync --group dev` resolves them natively. With pip, install each package manually.
- MyPy is strict: `disallow_untyped_defs = true`. All new functions need type annotations.
- Test directory is `tests/` (with an s).
