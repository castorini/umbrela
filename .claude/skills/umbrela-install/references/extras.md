# umbrela Optional Dependency Stacks

umbrela has four optional extras defined in `pyproject.toml`. The default install flow uses `cloud`.

## Extras

### `cloud` (default for dev setup)

API-based LLM backends for relevance assessment — lightweight, no GPU needed.

```bash
uv sync --extra cloud
pip install -e ".[cloud]"
```

| Package | Purpose |
|---------|---------|
| `google-cloud-aiplatform` | Google Vertex AI backend |
| `openai` | OpenAI/Azure API backend |
| `retry` | Retry logic for API calls |

### `hf`

HuggingFace transformers for local model inference.

```bash
uv sync --extra hf
pip install -e ".[hf]"
```

| Package | Purpose |
|---------|---------|
| `datasets` | HuggingFace dataset loading |
| `torch` | PyTorch backend |
| `transformers` | Model loading and inference |

### `fastchat`

FastChat-based model serving for local inference.

```bash
uv sync --extra fastchat
pip install -e ".[fastchat]"
```

| Package | Purpose |
|---------|---------|
| `fschat` | FastChat model serving |
| `torch` | PyTorch backend |
| `transformers` | Model loading |

### `pyserini`

Pyserini integration for retrieval in evaluation workflows.

```bash
uv sync --extra pyserini
pip install -e ".[pyserini]"
```

| Package | Purpose |
|---------|---------|
| `pyserini` | Lucene-based retrieval (requires Java 21) |

### `all`

Everything — union of all extras above.

```bash
uv sync --extra all
pip install -e ".[all]"
```

## Dev Dependencies (dependency-group)

| Package | Purpose |
|---------|---------|
| `mypy` | Static type checking (strict mode) |
| `pre-commit` | Git hook management |
| `pytest` | Test runner |
| `ruff` | Linter and formatter |
| `shtab` | Shell tab-completion generation |
| `types-PyYAML` | MyPy stubs |
| `types-tqdm` | MyPy stubs |

## Combining Extras

Multiple extras can be combined:

```bash
uv sync --group dev --extra cloud --extra hf
pip install -e ".[cloud,hf]"
```
