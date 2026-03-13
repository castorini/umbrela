# umBRELA

umBRELA is an open-source reproduction of the Bing RELevance Assessor for query-passage relevance labeling. It uses large language models to assign 4-level relevance judgments (`0` to `3`) to query-document pairs and supports both cloud-hosted and open-weight judges.

The package is built for information retrieval evaluation workflows: you can run a single judge, compare multiple judge backends, or generate modified qrels and downstream metrics from an existing retrieval run.

## What it includes

- `GPTJudge` for OpenAI or Azure OpenAI models
- `GeminiJudge` for Vertex AI Gemini models
- `HGFLLMJudge` for local Hugging Face transformer models
- `OSLLMJudge` for FastChat-compatible open models
- `EnsembleJudge` for majority-vote labeling across multiple judges

## Setup

`umbrela` uses [`uv`](https://docs.astral.sh/uv/) for Python environment management, dependency resolution, and command execution.

It is built on Python 3.11 (other versions might work, but YMMV).

### Install `uv`

Install `uv` with Astral's official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `uv` already works in your shell, you can skip this step. Otherwise, restart your shell or add `uv` to the current shell session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Prerequisites

- Install Java 21 once on the machine. `pyserini` uses Lucene and the JVM for qrel and passage access.

### Development installation

For development or the latest features, install from source with a repo-local virtual environment:

```bash
git clone https://github.com/castorini/umbrela.git
cd umbrela
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --group dev
```

If you prefer not to activate the virtual environment, run commands through `uv run`, for example:

```bash
uv run python examples/e2e.py --help
uv run umbrela --help
uv run pre-commit run --all-files
```

The repository pins the default interpreter to Python 3.11 with `.python-version`, so `uv` will also select that version automatically when possible.

Install only the backends you need:

```bash
# GPT + Gemini
uv sync --group dev --extra cloud

# Hugging Face local models
uv sync --group dev --extra hf

# FastChat local models
uv sync --group dev --extra fastchat

# Everything
uv sync --group dev --extra all
```

### Environment variables

Add a repo-local `.env` file with only the credentials for the backend you plan to run:

```dotenv
# GPTJudge via Azure OpenAI
OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_API_BASE=...
DEPLOYMENT_NAME=...

# GeminiJudge via Vertex AI
GCLOUD_PROJECT=...
GCLOUD_REGION=...

# HGFLLMJudge via Hugging Face
HF_TOKEN=...
HF_CACHE_DIR=...
```

Only set the variables required by the judge you are using.

### CLI usage

`umbrela` is the canonical command-line interface for this repository.
Use `uv run umbrela ...` if the virtual environment is not activated.

Direct minimal JSON input:

```bash
uv run umbrela judge \
  --backend gpt \
  --model gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --output json
```

Direct input from standard input:

```bash
echo '{"query":"anthropological definition of environment","candidates":["Environmental anthropology examines relationships between humans and their environment across space and time."]}' \
  | uv run umbrela judge --backend gpt --model gpt-4o --stdin --output json
```

Batch judging:

```bash
uv run umbrela judge \
  --backend gemini \
  --model gemini-1.5-pro \
  --input-file requests.jsonl \
  --output-file judgments.jsonl
```

Qrel-backed evaluation:

```bash
uv run umbrela evaluate \
  --backend gpt \
  --model gpt-4o \
  --qrel dl19-passage \
  --result-file run.trec \
  --prompt-type bing \
  --few-shot-count 0 \
  --output json
```

CLI introspection:

```bash
uv run umbrela describe judge --output json
uv run umbrela schema judge-direct-input
uv run umbrela validate judge \
  --input-json '{"query":"q","candidates":["p1","p2"]}' \
  --output json
uv run umbrela doctor --output json
```

Supported `--qrel` values: `dl19-passage`, `dl20-passage`, `dl21-passage`, `dl22-passage`, `dl23-passage`, `robust04`, `robust05`.

Supported prompt styles: `bing` (the Bing RELevance Assessor prompt) and
`basic`. Set `--few-shot-count 0` for zero-shot labeling and values greater
than `0` for few-shot labeling. Built-in prompts are stored as YAML templates
under `src/umbrela/prompts/prompt_templates/`, but they still render to the
exact same flat prompt string that earlier `.txt` assets produced. Custom
`--prompt-file` inputs should also be YAML templates.

Machine-readable output uses the shared `castorini.cli.v1` envelope. The
`judge` command includes Umbrela's legacy judgment list inside that envelope,
while `evaluate` includes artifact paths and evaluation metrics.

Migration examples:

```bash
# old
uv run umbrela-gpt --qrel dl19-passage --result_file run.trec --prompt_type bing --model gpt-4o --few_shot_count 0

# new
uv run umbrela evaluate --backend gpt --qrel dl19-passage --result-file run.trec --prompt-type bing --model gpt-4o --few-shot-count 0
```

### Quick end-to-end smoke test

For the default async-first OpenAI/Azure OpenAI example, run:

```bash
uv run python examples/e2e.py --model gpt-4o
```

Pass `--use_azure_openai` to target Azure OpenAI instead of the public OpenAI API. The example uses bounded request concurrency; tune it with `--max_concurrency`.

For the synchronous compatibility example, run:

```bash
uv run python examples/sync_e2e.py --judge gpt --model gpt-4o
```

`sync_e2e.py` also retains the multi-backend smoke-test flow for `gemini`, `hf`, and `os`.

### Programmatic usage

```python
import asyncio

from dotenv import load_dotenv
from umbrela.gpt_judge import GPTJudge

load_dotenv()

judge_gpt = GPTJudge(qrel="dl19-passage", prompt_type="bing", model_name="gpt-4o")
```

```python
input_dict = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {
            "doc": {
                "segment": "The life cycle of a flea can last anywhere from 20 days to an entire year. It depends on how long the flea remains in the dormant stage (eggs, larvae, pupa). Outside influences, such as weather, affect the flea cycle. A female flea can lay around 20 to 25 eggs in one day."
            },
            "docid": "4834547",
        },
    ]
}

judgments = asyncio.run(judge_gpt.async_judge(request_dict=input_dict))
```

The synchronous compatibility shim remains available:

```python
judgments = judge_gpt.judge(request_dict=input_dict)
```

## Evaluation outputs

When you run end-to-end evaluation, umBRELA writes generated artifacts into repo-local directories:

- `modified_qrels/` for LLM-generated qrels
- `conf_matrix/` for confusion-matrix visualizations

For MS MARCO v2 passage lookups, `scripts/download_msmarco.sh` downloads the required corpus files into `data/`.

## Reference

If you use umBRELA, please cite the following paper:

[[2406.06519] UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor](https://arxiv.org/abs/2406.06519)

<!-- {% raw %} -->
```
@ARTICLE{upadhyay2024umbrela,
  title   = {UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor},
  author  = {Shivani Upadhyay and Ronak Pradeep and Nandan Thakur and Nick Craswell and Jimmy Lin},
  year    = {2024},
  journal = {arXiv:2406.06519}
}
```
<!-- {% endraw %} -->


## Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
