from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    prompt: str
    answer: str | None = None
    image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
