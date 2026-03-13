# umbrela — Agent Guidelines

## Overview

umBRELA is a Python package for LLM-based relevance assessment of query-passage pairs, reproducing the Bing RELevance Assessor. It assigns 4-class relevance labels (0-3) to query-document pairs using various LLMs.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Additional setup notes:

- Install Java 21 once on the host. `pyserini` relies on Lucene and JVM access for passage lookup and qrel utilities.
- If `uv` is not already on `PATH` after installation, add it with `export PATH="$HOME/.local/bin:$PATH"` instead of sourcing shell-specific env scripts.
- Recommended development bootstrap:
  - `uv python install 3.11`
  - `uv venv --python 3.11`
  - `source .venv/bin/activate`
  - `uv sync --group dev`
- `uv venv` and `uv sync` create and manage the repo-local virtual environment at `.venv`.
- The published package supports Python 3.11 and newer via `pyproject.toml`.
- The repository already pins the interpreter via `.python-version`, so `uv` will use that version automatically and download it if needed.
- Use `uv sync --group dev --extra cloud`, `uv sync --group dev --extra hf`, `uv sync --group dev --extra fastchat`, or `uv sync --group dev --extra all` when you need contributor tooling plus a specific backend stack.
- If you prefer not to activate the virtual environment, run commands through `uv run`.
- Run `uv run pre-commit run --all-files` before committing local changes when you want to execute the full contributor quality gate manually.
- Add a repo-local `.env` with only the variables needed for the backend you plan to run (see Environment Variables below).

## Environment Variables

- `OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`, `DEPLOYMENT_NAME` — for `GPTJudge` (Azure preferred; falls back to plain OpenAI if Azure vars absent)
- `GCLOUD_PROJECT`, `GCLOUD_REGION` — for `GeminiJudge` (uses Vertex AI)
- `HF_TOKEN`, `HF_CACHE_DIR` — for `HGFLLMJudge`

## Running Evaluations

Each judge module is also a CLI entry point:

```bash
# GPT (OpenAI/Azure)
uv run umbrela-gpt --qrel dl19-passage --result_file <path> --prompt_type bing --model gpt-4o --few_shot_count 0

# Gemini (Vertex AI)
uv run umbrela-gemini --qrel dl19-passage --result_file <path> --prompt_type bing --model gemini-1.0-pro --few_shot_count 0

# Open-source via HuggingFace
uv run umbrela-hf --qrel dl19-passage --result_file <path> --prompt_type bing --model meta-llama/Llama-2-7b --few_shot_count 0 --device cuda

# Open-source via FastChat
uv run umbrela-os --qrel dl19-passage --result_file <path> --prompt_type bing --model lmsys/vicuna-7b-v1.5 --few_shot_count 0

# Ensemble (majority vote across multiple judges)
uv run umbrela-ensemble --qrel dl19-passage --result_file <path> --prompt_type bing \
  --llm_judges "GPTJudge,GeminiJudge" --model_names "gpt-4o,gemini-1.0-pro" --few_shot_count 0
```

Supported `--qrel` values: `dl19-passage`, `dl20-passage`, `dl21-passage`, `dl22-passage`, `dl23-passage`, `robust04`, `robust05`.

Prompt types: `bing` (default, mirrors Bing BRELA prompt) or `basic`. Combined with `--few_shot_count`: 0 = zeroshot, >0 = fewshot.

## Architecture

```
src/umbrela/
  llm_judge.py       # Abstract base class LLMJudge — shared evaluation logic
  gpt_judge.py       # GPTJudge: OpenAI/Azure OpenAI
  gemini_judge.py    # GeminiJudge: Google Vertex AI
  hgfllm_judge.py    # HGFLLMJudge: HuggingFace transformers (batched, DataLoader)
  osllm_judge.py     # OSLLMJudge: open-source via FastChat load_model
  ensemble_judge.py  # Runs multiple judges, applies majority vote
  prompts/           # Prompt templates: qrel_{zeroshot,fewshot}_{bing,basic}.txt
  utils/
    qrel_utils.py    # Qrel I/O, pyserini integration, passage retrieval, nDCG scoring
    common_utils.py  # Prompt generation, response parsing, qrel writing
src/eval/test.py     # Example usage snippet
```

### Key Design Patterns

- `LLMJudge` (abstract): `__init__` loads prompt template; subclasses implement `predict_with_llm()` and `judge()`. `evalute_results_with_qrel()` orchestrates the full pipeline: generate holes → call judge → write modified qrel → compute Cohen's kappa and nDCG.
- `judge()` returns a list of dicts with keys: `model`, `query`, `passage`, `prompt`, `prediction`, `judgment` (int 0-3), `result_status` (1 if parseable, 0 if fallback).
- Passage retrieval uses pyserini's `LuceneIndexReader` for MS MARCO v1 (dl19/dl20) and direct file access for MS MARCO v2 (dl21-23).
- Modified qrels are written to `modified_qrels/` directory; confusion matrices to `conf_matrix/`.
- Response parsing (`common_utils.parse_fewshot_response`) uses a ranked list of regex patterns to extract the 0-3 score; falls back to 0 if none match.

### Programmatic API

```python
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv
load_dotenv()

judge = GPTJudge(qrel="dl19-passage", prompt_type="bing", model_name="gpt-4o")
judgments = judge.judge(request_dict=input_dict)
# input_dict: {"query": {"text": ..., "qid": ...}, "candidates": [{"doc": {"segment": ...}, "docid": ...}]}
```
