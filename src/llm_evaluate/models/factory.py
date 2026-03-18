from __future__ import annotations

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.base import ModelAdapter
from llm_evaluate.models.openai_compatible_adapter import OpenAICompatibleAdapter
from llm_evaluate.models.transformers_adapter import TransformersAdapter


def build_model_adapter(config: ModelConfig) -> ModelAdapter:
    if config.backend == "transformers":
        return TransformersAdapter(config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleAdapter(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
