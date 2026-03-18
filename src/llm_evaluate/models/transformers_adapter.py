from __future__ import annotations

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.base import ModelAdapter


class TransformersAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig):
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers dependencies missing. Install with "
                "`pip install transformers torch pillow`."
            ) from exc

        self.config = config
        model_kwargs: dict = {"trust_remote_code": config.trust_remote_code}
        if config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = config.torch_dtype
        if config.device_map:
            model_kwargs["device_map"] = config.device_map

        pipeline_kwargs: dict = {
            "model": config.model_name_or_path,
            "model_kwargs": model_kwargs,
        }
        if config.device is not None:
            pipeline_kwargs["device"] = config.device

        if config.task_type == "llm":
            self.pipe = pipeline("text-generation", **pipeline_kwargs)
        else:
            self.pipe = pipeline("image-text-to-text", **pipeline_kwargs)

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
