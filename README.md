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

### Prerequisites

- Install Java 21 once on the machine. `pyserini` uses Lucene and the JVM for qrel and passage access.

### Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create/update this repository's .venv and install the cloud backends
uv sync --extra cloud
```

`uv sync` creates a repo-local virtual environment at `.venv`. The repository pins the default interpreter to Python 3.11 with `.python-version`, so `uv` will use that version automatically and download it if needed.

Install only the backends you need:

```bash
# GPT + Gemini
uv sync --extra cloud

# Hugging Face local models
uv sync --extra hf

# FastChat local models
uv sync --extra fastchat

# Everything
uv sync --extra all
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

Each judge has a short `uv run` entry point:

```bash
# GPT (OpenAI or Azure OpenAI)
uv run umbrela-gpt --qrel dl19-passage --result_file <path> --prompt_type bing --model gpt-4o --few_shot_count 0

# Gemini (Vertex AI)
uv run umbrela-gemini --qrel dl19-passage --result_file <path> --prompt_type bing --model gemini-1.0-pro --few_shot_count 0

# Open-source via Hugging Face transformers
uv run umbrela-hf --qrel dl19-passage --result_file <path> --prompt_type bing --model meta-llama/Llama-2-7b --few_shot_count 0 --device cuda

# Open-source via FastChat
uv run umbrela-os --qrel dl19-passage --result_file <path> --prompt_type bing --model lmsys/vicuna-7b-v1.5 --few_shot_count 0

# Ensemble
uv run umbrela-ensemble --qrel dl19-passage --result_file <path> --prompt_type bing \
  --llm_judges "GPTJudge,GeminiJudge" --model_names "gpt-4o,gemini-1.0-pro" --few_shot_count 0
```

Supported `--qrel` values: `dl19-passage`, `dl20-passage`, `dl21-passage`, `dl22-passage`, `dl23-passage`, `robust04`, `robust05`.

Supported prompt styles: `bing` (the Bing RELevance Assessor prompt) and `basic`. Set `--few_shot_count 0` for zero-shot labeling and values greater than `0` for few-shot labeling.

### Quick end-to-end smoke test

For a small in-memory request that mirrors the package API, run:

```bash
uv run python examples/e2e.py --judge gpt --model gpt-4o
```

You can swap `--judge` to `gemini`, `hf`, or `os` and pass the matching model identifier plus any backend-specific environment variables from the setup section.

### Programmatic usage

```python
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv

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
