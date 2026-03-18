from __future__ import annotations

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.base import ModelAdapter


class TransformersAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig):
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers dependencies missing. Install with `pip install transformers torch pillow`."
            ) from exc

        self.config = config
        if config.task_type == "llm":
            self.pipe = pipeline(
                "text-generation",
                model=config.model_name_or_path,
            )
        else:
            self.pipe = pipeline(
                "image-text-to-text",
                model=config.model_name_or_path,
            )

    def generate(self, prompt: str, image: str | None = None) -> str:
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.task_type == "llm":
            out = self.pipe(prompt, **kwargs)
            text = out[0].get("generated_text", "")
        else:
            payload = {"text": prompt}
            if image:
                payload["image"] = image
            out = self.pipe(payload, **kwargs)
            text = str(out[0].get("generated_text", out[0]))
        return text.strip()
