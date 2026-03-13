# Umbrela CLI For Agents

`umbrela ...` is the canonical command-line interface for this repository.
Use it in preference to the old backend-specific entry points. In an activated
environment, run `umbrela ...` directly. If the virtual environment is not
activated, the development fallback is `uv run umbrela ...`.

## Command Overview

- `umbrela judge`: Run a selected backend on direct JSON input or batch JSONL.
- `umbrela evaluate`: Run qrel-backed evaluation and emit artifact metadata.
- `umbrela describe`: Inspect command metadata and examples.
- `umbrela schema`: Print JSON schemas for direct input, batch input, outputs,
  and the shared CLI envelope.
- `umbrela validate`: Validate request payloads and file requirements without
  executing a model.
- `umbrela doctor`: Report Python, environment-variable, and qrel readiness.

## Minimal Direct Input

```bash
umbrela judge \
  --backend gpt \
  --model gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --output json
```

The direct form accepts lightweight caller payloads:

```json
{
  "query": "how long is life cycle of flea",
  "candidates": [
    "The life cycle of a flea can last anywhere from 20 days to an entire year."
  ]
}
```

The CLI normalizes that into Umbrela's canonical internal shape:
`query.text` plus `candidates[].doc.segment`. Caller-supplied `qid` and `docid`
remain optional.

## Batch And Evaluation Examples

```bash
umbrela judge \
  --backend gemini \
  --model gemini-1.5-pro \
  --input-file requests.jsonl \
  --output-file judgments.jsonl
```

```bash
umbrela evaluate \
  --backend gpt \
  --model gpt-4o \
  --qrel dl19-passage \
  --result-file run.trec \
  --output json
```

## Introspection Examples

```bash
umbrela describe judge --output json
umbrela schema judge-direct-input
umbrela validate judge \
  --input-json '{"query":"q","candidates":["p1","p2"]}' \
  --output json
umbrela doctor --output json
```

## Migration Examples

Old:

```bash
umbrela-gpt --qrel dl19-passage --result_file run.trec --prompt_type bing --model gpt-4o --few_shot_count 0
```

New:

```bash
umbrela evaluate --backend gpt --qrel dl19-passage --result-file run.trec --prompt-type bing --model gpt-4o --few-shot-count 0
```
