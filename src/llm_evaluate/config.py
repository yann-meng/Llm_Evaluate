from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class DatasetConfig:
    source: Literal["huggingface", "modelscope", "local"]
    dataset_id: str = "default_dataset"
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
    backend: Literal["transformers", "openai_compatible", "vllm", "sglang"]
    task_type: Literal["llm", "vlm"] = "llm"
    model_source: Literal["local", "huggingface", "modelscope"] = "huggingface"
    model_name_or_path: str = ""
    api_base: str | None = None
    api_key: str | None = None
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device: str | int | None = None
    device_map: str | None = None
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass(slots=True)
class MetricConfig:
    names: list[str] = field(default_factory=lambda: ["exact_match"])


@dataclass(slots=True)
class RunConfig:
    run_name: str
    output_dir: str
    model: ModelConfig
    dataset: DatasetConfig | None = None
    datasets: list[DatasetConfig] = field(default_factory=list)
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
    dataset = DatasetConfig(**content["dataset"]) if "dataset" in content else None
    datasets = [DatasetConfig(**item) for item in content.get("datasets", [])]
    if dataset and not datasets:
        datasets = [dataset]
    if not datasets:
        raise ValueError("Config requires `dataset` or `datasets`.")

    model = ModelConfig(**content["model"])
    metrics = MetricConfig(**content.get("metrics", {}))
    return RunConfig(
        run_name=content.get("run_name", "default_run"),
        output_dir=content.get("output_dir", "outputs"),
        batch_size=content.get("batch_size", 1),
        dataset=dataset,
        datasets=datasets,
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
        stack: list[tuple[int, dict | list]] = [(-1, result)]
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))

            while stack and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]
            stripped = line.strip()
            if stripped.startswith("- "):
                item_text = stripped[2:].strip()
                if not isinstance(parent, list):
                    raise ValueError("Invalid YAML list structure in fallback parser.")
                if ":" in item_text:
                    key, _, value = item_text.partition(":")
                    value = value.strip()
                    item = {key.strip(): _parse_yaml_scalar(value)}
                    parent.append(item)
                    stack.append((indent, item))
                else:
                    parent.append(_parse_yaml_scalar(item_text))
                continue

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                next_nonempty = ""
                for candidate in content.splitlines()[content.splitlines().index(raw_line) + 1 :]:
                    if candidate.strip() and not candidate.lstrip().startswith("#"):
                        next_nonempty = candidate.strip()
                        break
                container: dict | list
                container = [] if next_nonempty.startswith("- ") else {}
                if isinstance(parent, dict):
                    parent[key] = container
                else:
                    parent.append(container)
                stack.append((indent, container))
            elif isinstance(parent, dict):
                parent[key] = _parse_yaml_scalar(value)
            else:
                parent.append({key: _parse_yaml_scalar(value)})
        return result


def _parse_yaml_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit() and value.count(".") < 2:
        return float(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value.strip("'\"")
