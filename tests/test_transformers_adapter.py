import sys
import types

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.transformers_adapter import TransformersAdapter


def test_transformers_adapter_trust_remote_code_not_duplicated(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_pipeline(task: str, **kwargs):
        calls.append({"task": task, **kwargs})

        def _runner(prompt: str, **_gen_kwargs):
            return [{"generated_text": prompt}]

        return _runner

    fake_transformers = types.SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = ModelConfig(
        backend="transformers",
        task_type="llm",
        model_name_or_path="dummy-model",
        trust_remote_code=True,
    )

    adapter = TransformersAdapter(cfg)
    _ = adapter.generate("hello")

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["trust_remote_code"] is True
    assert "trust_remote_code" not in kwargs["model_kwargs"]
