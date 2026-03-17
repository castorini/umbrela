# Umbrela Input/Output Examples

## Judge Input (direct JSON)

```json
{
  "query": "how long is life cycle of flea",
  "candidates": [
    "The life cycle of a flea can last anywhere from 20 days to an entire year."
  ]
}
```

## Judge Input (batch JSONL)

Each line:

```json
{
  "query": {"text": "how long is life cycle of flea", "qid": "1037798"},
  "candidates": [
    {"doc": {"segment": "The life cycle of a flea can last anywhere from 20 days to an entire year."}, "docid": "7065735"}
  ]
}
```

Lightweight shorthand (candidates as plain strings):

```json
{
  "query": "how long is life cycle of flea",
  "candidates": ["The life cycle of a flea can last anywhere from 20 days to an entire year."]
}
```

## Judge Output (JSONL)

One record per query-candidate pair:

```json
{
  "model": "gpt-4o",
  "query": "how long is life cycle of flea",
  "passage": "The life cycle of a flea can last anywhere from 20 days to an entire year.",
  "prompt": "...",
  "prediction": "The passage directly answers the question about flea life cycle duration...\n\n##final score: 3",
  "judgment": 3,
  "result_status": 1
}
```

Fields:
- `judgment`: integer 0–3 (the relevance label)
- `result_status`: 1 if parsed successfully, 0 if fallback to 0
- `prediction`: raw LLM response text
- `prompt`: the rendered prompt sent to the LLM

With `--include-reasoning`:

```json
{
  "model": "gpt-4o",
  "query": "...",
  "passage": "...",
  "prompt": "...",
  "prediction": "...",
  "judgment": 3,
  "result_status": 1,
  "reasoning": "Step-by-step analysis of query-passage relevance..."
}
```

## Evaluate Output (text mode)

```
Original nDCG@10: 0.5432
Modified nDCG@10: 0.5678
Modified qrel: modified_qrels/dl19-passage_gpt-4o_01230_0_1.txt
Confusion matrix: conf_matrix/dl19-passage_gpt-4o_01230_0_1.png
```

## Evaluate Output (JSON mode)

```json
{
  "schema_version": "castorini.cli.v1",
  "repo": "umbrela",
  "command": "evaluate",
  "status": "success",
  "artifacts": [
    {"kind": "modified-qrel", "path": "modified_qrels/dl19-passage_gpt-4o_01230_0_1.txt"},
    {"kind": "confusion-matrix", "path": "conf_matrix/dl19-passage_gpt-4o_01230_0_1.png"}
  ],
  "metrics": {
    "original_ndcg_10": 0.5432,
    "modified_ndcg_10": 0.5678
  }
}
```

## Modified Qrel File Naming

```
modified_qrels/{qrel_basename}_{model_name}_{judge_cat}{few_shot_count}_{num_sample}.txt
```

Example: `modified_qrels/dl19-passage_gpt-4o_01230_0_1.txt`
- `dl19-passage`: base qrel name
- `gpt-4o`: model used
- `0123`: judged categories (all four)
- `0`: few-shot count
- `1`: number of samples per pair
