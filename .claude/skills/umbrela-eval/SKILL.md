---
name: umbrela-eval
description: Use when analyzing umbrela evaluation results — comparing nDCG@10 scores across backends, interpreting confusion matrices, computing kappa agreement, and comparing modified qrels against human judgments. Use after running evaluate to interpret results.
---

# Umbrela Eval

Analyze and compare umbrela evaluation results across backends, models, and configurations.

## When to Use

- After `umbrela evaluate` — interpret nDCG@10 scores and confusion matrices
- When comparing LLM judge backends (e.g., gpt-4o vs gemini-pro)
- When comparing prompt types (bing vs basic)
- When assessing agreement between LLM and human judgments

## What It Does

### Backend Comparison
- Compare nDCG@10 scores across different backends/models on the same qrel
- Show original vs modified nDCG@10 deltas
- Identify which backends agree most with human judgments

### Confusion Matrix Analysis
- Load confusion matrix PNGs from `conf_matrix/` directory
- Report per-category accuracy (0, 1, 2, 3)
- Identify systematic biases (e.g., model over-predicts category 2)

### Agreement Metrics
- Cohen's kappa between LLM labels and human labels
- Per-category precision, recall, F1
- Overall accuracy

## Usage

Compare evaluation results:
```bash
python3 .claude/skills/umbrela-eval/scripts/compare.py \
  --qrel dl19-passage \
  --run-a modified_qrels/dl19-passage_gpt-4o_01230_0_1.txt \
  --run-b modified_qrels/dl19-passage_gemini-pro_01230_0_1.txt
```

Or use the CLI directly:
```bash
# Run evaluation
umbrela evaluate --backend gpt --model gpt-4o \
  --qrel dl19-passage --result-file run.trec --output json

# View judgment artifacts
umbrela view judgments.jsonl --records 10
```

## Reference Files

- `references/qrels.md` — Standard qrels, nDCG@10 baselines, and evaluation conventions

## Comparison Script

See `scripts/compare.py` for the side-by-side comparison tool.

## Gotchas

- `evaluate` uses cached modified qrels by default. Use `--regenerate` to force re-judging.
- Confusion matrices compare LLM labels to human labels only for pairs where both exist — "holes" (new judgments) are not in the matrix.
- nDCG@10 differences between original and modified qrels reflect the impact of judging previously-unjudged documents, not just label agreement.
- `--judge-cat 2,3` means only relevant-category pairs are judged — this changes which holes are filled and affects nDCG differently than judging all categories.
- Ensemble results represent majority vote. Check individual backend confusion matrices for per-backend quality.
- pyserini is required for evaluation (`uv sync --extra pyserini`).
