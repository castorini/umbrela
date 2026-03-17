---
name: umbrela-quickstart
description: Use when working with umbrela CLI commands (judge, evaluate), backend selection (gpt, gemini, hf, os, ensemble), qrel handling (dl19-passage, dl20-passage, etc.), relevance labels (0–3), or introspection (doctor, describe, schema, validate). Covers all entry points, flags, and evaluation workflows.
---

# Umbrela Quickstart

Reference for the `umbrela` CLI — the tool for LLM-based relevance assessment of query-passage pairs, reproducing the Bing BRELA methodology with 4-class labels (0–3).

## CLI Entry Point

```bash
umbrela <command> [options]
```

## Primary Commands

| Command | Purpose |
|---------|---------|
| `judge` | Run a single backend on direct JSON or batch JSONL input |
| `evaluate` | Generate modified qrels and compute nDCG@10 from a result file |

## Introspection Commands

| Command | Purpose |
|---------|---------|
| `doctor` | Check Python version, API keys, backend readiness |
| `describe <cmd>` | Machine-readable command contract |
| `schema <name>` | Print JSON Schema for inputs/outputs |
| `validate <target>` | Validate inputs without running models |
| `prompt list\|show\|render` | Inspect and render prompt templates |
| `view <path>` | Inspect existing artifact files |

## Quick Workflow

```bash
# 1. Check environment
umbrela doctor

# 2. Judge query-passage pairs
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl

# 3. Evaluate against standard qrels
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec
```

## Reference Files

Read these on demand for details:

- `references/cli-examples.md` — Common invocations for each command
- `references/input-output-examples.md` — JSONL format examples and judgment output
- `references/backends.md` — Backend selection guide and provider configuration
- `references/qrel-handling.md` — Standard qrels, label meanings, and evaluation workflow

## Key Concepts

- **Labels 0–3**: 0=irrelevant, 1=related but doesn't answer, 2=partially answers, 3=perfectly answers
- **Backends**: `gpt` (OpenAI/Azure/OpenRouter), `gemini` (Vertex AI), `hf` (HuggingFace), `os` (FastChat), `ensemble` (majority voting, evaluate only)
- **Prompt types**: `bing` (BRELA-style, default) or `basic`
- **Modified qrels**: LLM judgments merged into standard qrels to fill "holes" (unjudged pairs)

## Gotchas

- `--backend ensemble` is **only valid for `evaluate`**, not `judge`.
- `--execution-mode async` is **only supported for GPT backend**.
- The `evaluate` command requires pyserini for qrel loading and passage retrieval. Install with `uv sync --extra pyserini`.
- `evaluate` writes two artifacts: a modified qrel file in `modified_qrels/` and a confusion matrix PNG in `conf_matrix/`.
- `--judge-cat` controls which relevance categories to evaluate. Default `0,1,2,3` judges all pairs. Use `2,3` to focus on relevant documents only.
- `--min-judgment` on `judge` filters output — pairs below the threshold go to `--filtered-output-file`.
- `--few-shot-count` controls few-shot examples in the prompt. Examples are drawn from the qrel dataset. Default is 0 (zero-shot).
- Config file (`.umbrela.toml` or `~/.config/umbrela/config.toml`) can set defaults.
