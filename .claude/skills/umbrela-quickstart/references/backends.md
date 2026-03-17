# Umbrela Backend Selection Guide

## Available Backends

| Backend | CLI value | Class | Provider | Required Env |
|---------|-----------|-------|----------|-------------|
| GPT | `gpt` | `GPTJudge` | OpenAI / Azure / OpenRouter | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` or `AZURE_OPENAI_*` |
| Gemini | `gemini` | `GeminiJudge` | Google Vertex AI | `GCLOUD_PROJECT`, `GCLOUD_REGION` |
| HuggingFace | `hf` | `HGFLLMJudge` | Local HF transformers | `HF_TOKEN` (optional) |
| Open-source | `os` | `OSLLMJudge` | FastChat server | None (local API) |
| Ensemble | `ensemble` | (orchestrated) | Multiple judges | Depends on constituent backends |

## Backend Selection

```bash
# GPT (default provider: OpenAI)
umbrela judge --backend gpt --model gpt-4o ...

# GPT via Azure
umbrela judge --backend gpt --model gpt-4o --use-azure-openai ...

# GPT via OpenRouter
umbrela judge --backend gpt --model openai/gpt-4o-mini --use-openrouter ...

# Gemini
umbrela judge --backend gemini --model gemini-pro ...

# HuggingFace (local)
umbrela judge --backend hf --model meta-llama/Llama-3-8B --device cuda ...

# Open-source via FastChat
umbrela judge --backend os --model vicuna-7b --device cuda ...
```

## Ensemble (evaluate only)

Ensemble combines multiple judges using majority voting:

```bash
umbrela evaluate --backend ensemble \
  --llm-judges GPTJudge,GeminiJudge \
  --model-names gpt-4o,gemini-pro \
  --qrel dl19-passage --result-file run.trec
```

- `--llm-judges`: comma-separated class names (e.g., `GPTJudge`, `GeminiJudge`, `HGFLLMJudge`, `OSLLMJudge`)
- `--model-names`: comma-separated model identifiers aligned with `--llm-judges`
- Ensemble is **not available** for `judge` command — only `evaluate`

## Provider Detection

OpenRouter is auto-detected when:
1. `--use-openrouter` flag is set, or
2. Model name starts with `openrouter/` prefix, or
3. `OPENROUTER_API_KEY` is set and `OPENAI_API_KEY` is absent

## Async Support

Only `gpt` backend supports `--execution-mode async`. Other backends run synchronously.

## Environment Variables

| Variable | Backend | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | gpt | OpenAI API key |
| `OPENROUTER_API_KEY` | gpt | OpenRouter API key |
| `AZURE_OPENAI_API_BASE` | gpt | Azure endpoint |
| `AZURE_OPENAI_API_VERSION` | gpt | Azure API version |
| `AZURE_OPENAI_API_KEY` | gpt | Azure API key |
| `GCLOUD_PROJECT` | gemini | Google Cloud project ID |
| `GCLOUD_REGION` | gemini | Google Cloud region |
| `HF_TOKEN` | hf | HuggingFace token (optional) |
| `HF_CACHE_DIR` | hf | Model cache directory |
