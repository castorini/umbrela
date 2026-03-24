from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration


SAMPLE_REQUEST: dict[str, Any] = {
    "query": {"text": "how long is life cycle of flea", "qid": "264014"},
    "candidates": [
        {"doc": {"segment": "first passage"}, "docid": "d1"},
        {"doc": {"segment": "second passage"}, "docid": "d2"},
    ],
}


def _write_prompt_file(tmp_path: Path) -> str:
    prompt_file = tmp_path / "prompt.yaml"
    prompt_file.write_text(
        'method: "custom"\n'
        'system_message: ""\n'
        'prefix_user: "{examples}\\nQuery: {query}\\nPassage: {passage}\\n"\n',
        encoding="utf-8",
    )
    return str(prompt_file)


def _install_fake_optional_modules(monkeypatch: Any) -> None:
    def set_module(name: str, module: types.ModuleType) -> None:
        monkeypatch.setitem(sys.modules, name, module)

    fake_vertexai = types.ModuleType("vertexai")
    setattr(fake_vertexai, "init", lambda **kwargs: None)
    fake_gen = types.ModuleType("vertexai.generative_models")

    class GenerationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class GenerativeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate_content(self, prompt: str, generation_config: Any = None) -> Any:
            del prompt, generation_config
            return types.SimpleNamespace(text="##final score: 3")

    setattr(fake_gen, "GenerationConfig", GenerationConfig)
    setattr(fake_gen, "GenerativeModel", GenerativeModel)
    set_module("vertexai", fake_vertexai)
    set_module("vertexai.generative_models", fake_gen)

    fake_retry = types.ModuleType("retry")
    setattr(fake_retry, "retry", lambda *args, **kwargs: (lambda fn: fn))
    set_module("retry", fake_retry)

    fake_datasets = types.ModuleType("datasets")
    setattr(
        fake_datasets, "Dataset", types.SimpleNamespace(from_list=lambda rows: rows)
    )
    set_module("datasets", fake_datasets)

    fake_torch = types.ModuleType("torch")
    setattr(fake_torch, "cuda", types.SimpleNamespace(is_available=lambda: False))
    setattr(fake_torch, "tensor", lambda value: value)
    set_module("torch", fake_torch)
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    setattr(fake_torch_utils_data, "DataLoader", list)
    set_module("torch.utils", fake_torch_utils)
    set_module("torch.utils.data", fake_torch_utils_data)

    fake_transformers = types.ModuleType("transformers")
    setattr(
        fake_transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    setattr(
        fake_transformers,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    setattr(fake_transformers, "DataCollatorWithPadding", object)
    set_module("transformers", fake_transformers)

    fake_fastchat = types.ModuleType("fastchat")
    fake_fastchat_model = types.ModuleType("fastchat.model")
    setattr(
        fake_fastchat_model, "load_model", lambda *args, **kwargs: (object(), object())
    )
    set_module("fastchat", fake_fastchat)
    set_module("fastchat.model", fake_fastchat_model)

    fake_generation = types.ModuleType("transformers.generation")

    class FakeGenerationConfig:
        @classmethod
        def from_model_config(cls, config: Any) -> Any:
            del config
            return types.SimpleNamespace(max_new_tokens=0, do_sample=False)

    setattr(fake_generation, "GenerationConfig", FakeGenerationConfig)
    set_module("transformers.generation", fake_generation)


def _reload_module(name: str) -> Any:
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_gpt_judge_reports_missing_provider_configuration(monkeypatch: Any) -> None:
    fake_openai = types.ModuleType("openai")
    setattr(fake_openai, "BadRequestError", RuntimeError)
    setattr(
        fake_openai, "AsyncOpenAI", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    setattr(
        fake_openai,
        "AsyncAzureOpenAI",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    from umbrela.gpt_judge import GPTJudge

    with pytest.raises(KeyError):
        GPTJudge(
            qrel="dl19-passage",
            model_name="gpt-4o",
            prompt_type="bing",
            few_shot_count=0,
        )


def test_gemini_predict_with_llm_falls_back_to_empty_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _install_fake_optional_modules(monkeypatch)
    gemini_module = _reload_module("umbrela.gemini_judge")
    GeminiJudge = gemini_module.GeminiJudge

    monkeypatch.setenv("GCLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GCLOUD_REGION", "us-central1")

    judge = GeminiJudge(
        qrel="dl19-passage",
        model_name="gemini-1.5-pro",
        prompt_file=_write_prompt_file(tmp_path),
        prompt_type=None,
        few_shot_count=0,
    )

    class BrokenClient:
        def generate_content(self, prompt: str, generation_config: Any = None) -> Any:
            del prompt, generation_config
            raise RuntimeError("boom")

    judge.client = BrokenClient()
    outputs = judge.predict_with_llm(SAMPLE_REQUEST, max_new_tokens=32, preprocess=True)

    assert outputs == ["", ""]
    judgments = judge.judge(SAMPLE_REQUEST, max_new_tokens=32, preprocess=True)
    assert [item["judgment"] for item in judgments] == [0, 0]
    assert [item["result_status"] for item in judgments] == [0, 0]


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("umbrela.gemini_judge", "GeminiJudge"),
        ("umbrela.hgfllm_judge", "HGFLLMJudge"),
        ("umbrela.osllm_judge", "OSLLMJudge"),
    ],
)
def test_optional_backends_share_judgment_contract(
    tmp_path: Path,
    monkeypatch: Any,
    module_name: str,
    class_name: str,
) -> None:
    _install_fake_optional_modules(monkeypatch)
    module = _reload_module(module_name)
    backend_cls = getattr(module, class_name)

    kwargs: dict[str, Any] = {
        "qrel": "dl19-passage",
        "model_name": "demo-model",
        "prompt_file": _write_prompt_file(tmp_path),
        "prompt_type": None,
        "few_shot_count": 0,
    }
    if class_name in {"HGFLLMJudge", "OSLLMJudge"}:
        kwargs["device"] = "cpu"

    if class_name == "GeminiJudge":
        monkeypatch.setenv("GCLOUD_PROJECT", "demo-project")
        monkeypatch.setenv("GCLOUD_REGION", "us-central1")

    judge = backend_cls(**kwargs)

    def fake_predict(
        request_dict: dict[str, Any], max_new_tokens: int, preprocess: bool
    ) -> list[str]:
        judge.prepare_request_inputs(request_dict, preprocess)
        del max_new_tokens
        return ["final score: 2", "unparseable output"]

    monkeypatch.setattr(judge, "predict_with_llm", fake_predict)
    judgments = judge.judge(SAMPLE_REQUEST, max_new_tokens=48, preprocess=True)

    assert [item["judgment"] for item in judgments] == [2, 0]
    assert [item["result_status"] for item in judgments] == [1, 0]
    assert judgments[0]["query"] == SAMPLE_REQUEST["query"]["text"]
    assert judgments[1]["passage"] == "second passage"
    assert "Query:" in judgments[0]["prompt"]
