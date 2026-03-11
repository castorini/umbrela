# umBRELA

## Instructions

`umbrela` now uses [`uv`](https://docs.astral.sh/uv/) for Python environment management, dependency resolution, and command execution.

### Prerequisites

- Install `uv`.
- Install Java 21 once on the machine. `pyserini` uses Lucene and the JVM for qrel and passage access.
- Install Maven if your local `pyserini` workflow expects it.

### Quick start

```bash
uv python install 3.12
uv sync --extra cloud
```

`uv` will pick Python 3.12 automatically because the repository now includes `.python-version`.

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

Add a `.env` file with the credentials for the backends you plan to run.

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

### Judgment generation snippet

#### Setting up the model judge:
```python
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv

load_dotenv()

judge_gpt = GPTJudge(qrel="dl19-passage", prompt_type="bing", model_name="gpt-4o")
```

#### Passing qrel-passages for evaluations:
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

## ✨ References

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


## 🙏 Acknowledgments

This research is supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.
