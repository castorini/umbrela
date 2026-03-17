# Umbrela Qrels Reference

## Standard Qrel Sets

| Name | Track | Queries | Passage Collection | Typical nDCG@10 Range |
|------|-------|---------|--------------------|-----------------------|
| `dl19-passage` | TREC DL 2019 | 43 | MS MARCO v1 | 0.50–0.75 |
| `dl20-passage` | TREC DL 2020 | 54 | MS MARCO v1 | 0.45–0.70 |
| `dl21-passage` | TREC DL 2021 | 53 | MS MARCO v2 | 0.40–0.65 |
| `dl22-passage` | TREC DL 2022 | 76 | MS MARCO v2 | 0.35–0.60 |
| `dl23-passage` | TREC DL 2023 | 82 | MS MARCO v2 | 0.35–0.60 |
| `robust04` | Robust 2004 | 249 | TREC Disks 4&5 | 0.30–0.50 |
| `robust05` | Robust 2005 | 50 | AQUAINT | 0.25–0.45 |

nDCG@10 ranges are approximate and depend on the retrieval system being evaluated.

## Evaluation Workflow

```
Original qrel (human labels)
    │
    ├── Identify "holes" (unjudged query-doc pairs in result file)
    │
    ▼
LLM judge fills holes → Modified qrel
    │
    ├── Compute nDCG@10 (original qrel)
    ├── Compute nDCG@10 (modified qrel)
    └── Generate confusion matrix (LLM vs human, overlapping pairs)
```

## Modified Qrel Naming Convention

```
modified_qrels/{qrel}_{model}_{judge_cat}{few_shot}_{num_sample}.txt
```

Example: `modified_qrels/dl19-passage_gpt-4o_01230_0_1.txt`
- `dl19-passage`: base qrel
- `gpt-4o`: judge model
- `0123`: all four categories judged
- `0`: zero-shot (no few-shot examples)
- `1`: one sample per pair

## Interpreting nDCG@10 Differences

| Δ nDCG@10 | Interpretation |
|-----------|----------------|
| < 0.01 | Negligible — holes had minimal impact |
| 0.01–0.05 | Moderate — some unjudged relevant docs found |
| > 0.05 | Significant — many unjudged docs were relevant |
| Negative | LLM labels disagree with human labels on key docs |

## Cohen's Kappa Guidelines

| Kappa | Agreement Level |
|-------|-----------------|
| < 0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| > 0.80 | Almost perfect |

For 4-class relevance (0–3), kappa values above 0.4 are typical for LLM judges vs human annotations.

## Confusion Matrix Location

Generated at: `conf_matrix/{qrel}_{model}_{judge_cat}{few_shot}_{num_sample}.png`

The matrix shows:
- Rows: human labels (ground truth)
- Columns: LLM predicted labels
- Diagonal: agreements
- Off-diagonal: disagreements (systematic biases visible as patterns)
