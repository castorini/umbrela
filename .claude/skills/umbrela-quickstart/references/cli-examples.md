# Umbrela CLI Examples

## judge

```bash
# Batch judging from JSONL
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl

# Direct input (single query, stdout)
umbrela judge --backend gpt --model gpt-4o \
  --input-json '{"query":"how long is life cycle of flea","candidates":["The life cycle of a flea can last anywhere from 20 days to an entire year."]}' \
  --output json

# OpenRouter backend
umbrela judge --backend gpt --model openai/gpt-4o-mini \
  --use-openrouter --input-file pairs.jsonl --output-file judgments.jsonl

# Azure OpenAI
umbrela judge --backend gpt --model gpt-4o \
  --use-azure-openai --input-file pairs.jsonl --output-file judgments.jsonl

# Gemini backend
umbrela judge --backend gemini --model gemini-pro \
  --input-file pairs.jsonl --output-file judgments.jsonl

# With filtering (only keep judgments >= 2)
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl \
  --min-judgment 2 --filtered-output-file filtered.jsonl

# Few-shot prompting (requires qrel for examples)
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl \
  --few-shot-count 4

# Async execution (GPT only)
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl \
  --execution-mode async --max-concurrency 16

# With reasoning fields
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl --include-reasoning

# Dry run
umbrela judge --backend gpt --model gpt-4o \
  --input-file pairs.jsonl --output-file judgments.jsonl --dry-run

# Bing prompt type (default)
umbrela judge --backend gpt --model gpt-4o --prompt-type bing --input-file pairs.jsonl --output-file judgments.jsonl

# Basic prompt type
umbrela judge --backend gpt --model gpt-4o --prompt-type basic --input-file pairs.jsonl --output-file judgments.jsonl
```

## evaluate

```bash
# Evaluate against DL19
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec

# Evaluate with specific categories
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --judge-cat 2,3

# Ensemble evaluation
umbrela evaluate --backend ensemble \
  --llm-judges GPTJudge,GeminiJudge --model-names gpt-4o,gemini-pro \
  --qrel dl19-passage --result-file run.trec

# Regenerate cached qrels
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --regenerate

# Few-shot evaluation
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --few-shot-count 4

# Dry run (validate prerequisites)
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --dry-run

# JSON output
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --output json
```

## Introspection

```bash
# Environment check
umbrela doctor
umbrela doctor --output json

# Command contract
umbrela describe judge --output json
umbrela describe evaluate --output json

# JSON Schemas
umbrela schema judge-direct-input
umbrela schema judge-batch-input-record

# Prompt inspection
umbrela prompt list
umbrela prompt show --prompt-type bing
umbrela prompt show --prompt-type basic --few-shot-count 4
umbrela prompt render --prompt-type bing \
  --input-json '{"query":"test query","candidates":["test passage"]}' --part user

# View artifacts
umbrela view judgments.jsonl --records 5

# Validate inputs
umbrela validate judge --input-file pairs.jsonl
umbrela validate evaluate --qrel dl19-passage --result-file run.trec
```
