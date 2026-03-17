---
name: umbrela-verify
description: Use when validating umbrela judge outputs — checks label range (0–3), qid/docid completeness, result_status consistency, backend metadata, and JSONL integrity. Wraps `umbrela validate` plus custom assertions. Use after running judge or evaluate to verify output correctness.
---

# Umbrela Verify

Validates umbrela judgment outputs for label correctness, completeness, and consistency.

## When to Use

- After `umbrela judge` — verify judgment output integrity
- After `umbrela evaluate` — verify modified qrel completeness
- Before using judgments for downstream analysis or comparison
- When comparing judgments across backends or models

## What It Checks

### JSONL Integrity
- Every line is valid JSON
- No trailing commas, no truncated records

### Judge Output
- Every record has `model`, `query`, `passage`, `judgment`, and `result_status`
- `judgment` is integer 0–3
- `result_status` is 0 or 1 (1 = successfully parsed, 0 = fallback)
- No duplicate query-passage pairs
- All records use the same model (consistency check)

### Parse Success Rate
- Reports the fraction of records with `result_status == 1`
- Warns if parse failure rate exceeds 10%

### Modified Qrel (evaluate output)
- File exists in `modified_qrels/` directory
- Standard TREC qrel format (qid Q0 docid label)
- Labels are integers in expected range

## Usage

Run the verification script:

```bash
bash .claude/skills/umbrela-verify/scripts/verify.sh <artifact-path> [judge|qrel]
```

Or use the built-in validator first:

```bash
umbrela validate judge --input-file pairs.jsonl
umbrela validate evaluate --qrel dl19-passage --result-file run.trec
```

## Verification Script

See `scripts/verify.sh` for the runnable verification wrapper.

## Gotchas

- `umbrela validate` checks *input* contracts. The verify script checks *output* artifacts.
- `result_status == 0` means the LLM response couldn't be parsed into a 0–3 label — it falls back to 0. A high rate of `result_status == 0` suggests prompt issues or model incompatibility.
- The `prediction` field contains raw LLM text. The label is extracted by 39 regex patterns in `common_utils.py`. Check this if judgment distribution looks anomalous.
- Ensemble evaluate produces one qrel file — constituent backend outputs are not saved individually.
- Modified qrel naming: `{qrel}_{model}_{judge_cat}{few_shot}_{num_sample}.txt` — verify the filename matches expectations.
