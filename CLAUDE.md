# umbrela — Agent Guidelines

## Overview

umBRELA is a Python package for LLM-based relevance assessment of query-passage pairs, reproducing the Bing RELevance Assessor. It assigns 4-class relevance labels (0-3) to query-document pairs using various LLMs.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Additional setup notes:

- Install Java 21 only when you need qrel-backed evaluation or other
  `pyserini`-dependent workflows. Plain direct judging does not require Java.
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
- Run `bash scripts/quality_gate.sh commit` before committing local changes when you want to execute the full contributor quality gate manually. Use `bash scripts/quality_gate.sh push` for the non-mutating equivalent. The gate assumes `uv sync --group dev --extra cloud` has been run so MyPy can see the optional cloud backend imports.
- Add a repo-local `.env` with only the variables needed for the backend you plan to run (see Environment Variables below).
- Keep release-note updates in `docs/release-notes/` for user-visible changes.

## Environment Variables

- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`, `DEPLOYMENT_NAME` — for `GPTJudge` (public OpenAI by default, OpenRouter fallback when the OpenAI key is absent, Azure OpenAI only when explicitly requested)
- `GCLOUD_PROJECT`, `GCLOUD_REGION` — for `GeminiJudge` (uses Vertex AI)
- `HF_TOKEN`, `HF_CACHE_DIR` — for `HGFLLMJudge`

## Running Evaluations

Use the packaged `umbrela` CLI instead of backend-specific entry points.
Use plain `umbrela ...` in an activated environment and `uv run umbrela ...`
only as the fallback when the virtual environment is not activated:

```bash
# Direct minimal JSON input
umbrela judge \
  --backend gpt \
  --model openai/gpt-4o-mini \
  --use-openrouter \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --output json

# Batch evaluation against a qrel and retrieval run
umbrela evaluate \
  --backend gpt \
  --model gpt-4o \
  --qrel dl19-passage \
  --result-file <path> \
  --prompt-type bing \
  --few-shot-count 0 \
  --output json

# Introspection and validation
umbrela describe judge --output json
umbrela schema judge-direct-input
umbrela doctor --output json
```

Supported `--qrel` values: `dl19-passage`, `dl20-passage`, `dl21-passage`, `dl22-passage`, `dl23-passage`, `robust04`, `robust05`.

Prompt types: `bing` (default, mirrors Bing BRELA prompt) or `basic`.
Combined with `--few-shot-count`: 0 = zero-shot, values greater than 0 =
few-shot.

## Testing

- Test tiers:
  - `core`: `uv sync --group dev --extra cloud && uv run pytest -q -m core tests`
  - `integration`: `uv sync --group dev --extra cloud && uv run pytest -q -m integration tests`
  - `live`: opt-in smoke tests such as `UMBRELA_LIVE_OPENAI_SMOKE=1 uv run pytest -q tests/test_live_openai_smoke.py`
- The quality gate order is Ruff, then core tests, then integration tests, then MyPy. The local hooks and `scripts/quality_gate.sh` script enforce that order in both pre-commit and pre-push stages.
- Keep `core` and `integration` coverage offline and deterministic.
- Optional dependency stacks (`cloud`, `hf`, `pyserini`) should remain smoke-testable in CI without live provider calls.
- Apply the shared pytest markers `core`, `integration`, and `live` at the module level when adding or moving tests.

## Architecture

```
src/umbrela/
  llm_judge.py       # Abstract base class LLMJudge — shared evaluation logic
  gpt_judge.py       # GPTJudge: OpenAI/OpenRouter/Azure OpenAI
  gemini_judge.py    # GeminiJudge: Google Vertex AI
  hgfllm_judge.py    # HGFLLMJudge: HuggingFace transformers (batched, DataLoader)
  osllm_judge.py     # OSLLMJudge: open-source via FastChat load_model
  ensemble_judge.py  # Runs multiple judges, applies majority vote
  prompts/           # YAML prompt templates + loader; runtime prompt surface stays a single flat string
  utils/
    qrel_utils.py    # Qrel I/O, pyserini integration, passage retrieval, nDCG scoring
    common_utils.py  # Request preprocessing, response parsing, qrel writing
src/eval/test.py     # Example usage snippet
```

### Key Design Patterns

- `LLMJudge` (abstract): `__init__` resolves a YAML-backed prompt template and preserves the historical single-string prompt surface; subclasses implement `predict_with_llm()` and `judge()`. `evalute_results_with_qrel()` orchestrates the full pipeline: generate holes → call judge → write modified qrel → compute Cohen's kappa and nDCG.
- `judge()` returns a list of dicts with keys: `model`, `query`, `passage`, `prompt`, `prediction`, `judgment` (int 0-3), `result_status` (1 if parseable, 0 if fallback).
- Passage retrieval uses pyserini's `LuceneIndexReader` for MS MARCO v1 (dl19/dl20) and direct file access for MS MARCO v2 (dl21-23).
- Modified qrels are written to `modified_qrels/` directory; confusion matrices to `conf_matrix/`.
- Response parsing (`common_utils.parse_fewshot_response`) uses a ranked list of regex patterns to extract the 0-3 score; falls back to 0 if none match.
- If a change affects prompt semantics, parsing behavior, backend defaults, or evaluation artifacts, document the migration path in the release note.

### Programmatic API

```python
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv
load_dotenv()

judge = GPTJudge(qrel="dl19-passage", prompt_type="bing", model_name="gpt-4o")
openrouter_judge = GPTJudge(
    qrel="dl19-passage",
    prompt_type="bing",
    model_name="anthropic/claude-3.5-sonnet",
    use_openrouter=True,
)
judgments = judge.judge(request_dict=input_dict)
# input_dict: {"query": {"text": ..., "qid": ...}, "candidates": [{"doc": {"segment": ...}, "docid": ...}]}
```
