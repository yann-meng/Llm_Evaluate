from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: str | None = None) -> str:
        raise NotImplementedError
