from __future__ import annotations

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.openai_compatible_adapter import OpenAICompatibleAdapter


class SGLangAdapter(OpenAICompatibleAdapter):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
