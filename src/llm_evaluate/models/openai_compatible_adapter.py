from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib import request

from llm_evaluate.config import ModelConfig
from llm_evaluate.models.base import ModelAdapter


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig):
        if not config.api_base:
            raise ValueError("openai_compatible backend requires api_base")
        self.config = config

    def generate(self, prompt: str, image: str | None = None) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        messages = [{"role": "user", "content": prompt}]
        if self.config.task_type == "vlm" and image:
            image_url = image
            local_image = Path(image)
            if local_image.exists():
                b64 = base64.b64encode(local_image.read_bytes()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{b64}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

        payload = {
            "model": self.config.model_name_or_path,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }

        req = request.Request(
            url=url,
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
