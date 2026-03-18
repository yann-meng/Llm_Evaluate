from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class DatasetConfig:
    source: Literal["huggingface", "modelscope", "local"]
    name: str | None = None
    subset: str | None = None
    split: str = "test"
    path: str | None = None
    input_column: str = "prompt"
    answer_column: str = "answer"
    image_column: str | None = None
    limit: int | None = None


@dataclass(slots=True)
class ModelConfig:
    backend: Literal["transformers", "openai_compatible"]
    task_type: Literal["llm", "vlm"] = "llm"
    model_name_or_path: str = ""
    api_base: str | None = None
    api_key: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass(slots=True)
class MetricConfig:
    names: list[str] = field(default_factory=lambda: ["exact_match"])


@dataclass(slots=True)
class RunConfig:
    run_name: str
    output_dir: str
    dataset: DatasetConfig
    model: ModelConfig
    metrics: MetricConfig = field(default_factory=MetricConfig)
    batch_size: int = 1

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    content = _load_yaml_with_fallback(raw_text)
    return _build_run_config(content)


def _build_run_config(content: dict[str, Any]) -> RunConfig:
    dataset = DatasetConfig(**content["dataset"])
    model = ModelConfig(**content["model"])
    metrics = MetricConfig(**content.get("metrics", {}))
    return RunConfig(
        run_name=content.get("run_name", "default_run"),
        output_dir=content.get("output_dir", "outputs"),
        batch_size=content.get("batch_size", 1),
        dataset=dataset,
        model=model,
        metrics=metrics,
    )


def _load_yaml_with_fallback(content: str) -> dict:
    try:
        import yaml

        return yaml.safe_load(content)
    except ModuleNotFoundError:
        # Minimal YAML subset parser for offline environments.
        result: dict = {}
        stack: list[tuple[int, dict]] = [(-1, result)]
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, _, value = line.strip().partition(":")
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                parent[key] = {}
                stack.append((indent, parent[key]))
                continue
            if value.startswith("[") and value.endswith("]"):
                parent[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            elif value.isdigit():
                parent[key] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                parent[key] = float(value)
            elif value.lower() in {"true", "false"}:
                parent[key] = value.lower() == "true"
            else:
                parent[key] = value.strip("'\"")
        return result
