from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm_evaluate.config import DatasetConfig
from llm_evaluate.data.schema import EvalSample


class DatasetLoadError(RuntimeError):
    pass


class DatasetLoader:
    def load(self, config: DatasetConfig) -> list[EvalSample]:
        if config.source == "huggingface":
            rows = self._load_huggingface(config)
        elif config.source == "modelscope":
            rows = self._load_modelscope(config)
        else:
            rows = self._load_local(config)
        return self._map_rows(rows, config)

    def _load_huggingface(self, config: DatasetConfig) -> list[dict[str, Any]]:
        if not config.name:
            raise DatasetLoadError("Hugging Face source requires dataset name")
        try:
            from datasets import load_dataset
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(
                "datasets dependency missing. Install with `pip install datasets`."
            ) from exc
        ds = load_dataset(config.name, config.subset, split=config.split)
        if config.limit:
            ds = ds.select(range(min(config.limit, len(ds))))
        return [dict(r) for r in ds]

    def _load_modelscope(self, config: DatasetConfig) -> list[dict[str, Any]]:
        if not config.name:
            raise DatasetLoadError("ModelScope source requires dataset name")
        try:
            from modelscope.msdatasets import MsDataset
        except Exception as exc:  # noqa: BLE001
            raise DatasetLoadError(
                "ModelScope dependency missing. Install with `pip install modelscope`."
            ) from exc

        ms_ds = MsDataset.load(config.name, subset_name=config.subset, split=config.split)
        rows = [dict(item) for item in ms_ds]
        if config.limit:
            rows = rows[: config.limit]
        return rows

    def _load_local(self, config: DatasetConfig) -> list[dict[str, Any]]:
        if not config.path:
            raise DatasetLoadError("Local source requires file path")
        path = Path(config.path)
        if not path.exists():
            raise DatasetLoadError(f"Local dataset not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            raise DatasetLoadError("Local dataset only supports .jsonl or .csv")

        if config.limit:
            rows = rows[: config.limit]
        return rows

    @staticmethod
    def _map_rows(rows: list[dict[str, Any]], config: DatasetConfig) -> list[EvalSample]:
        samples: list[EvalSample] = []
        reserved = {config.input_column, config.answer_column}
        if config.image_column:
            reserved.add(config.image_column)

        for idx, row in enumerate(rows):
            sample = EvalSample(
                sample_id=str(row.get("id", idx)),
                prompt=str(row.get(config.input_column, "")),
                answer=_extract_answer(row.get(config.answer_column)),
                image=_optional_to_str(row.get(config.image_column)) if config.image_column else None,
                metadata={k: v for k, v in row.items() if k not in reserved},
            )
            samples.append(sample)
        return samples


def _optional_to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _extract_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, list) and text:
            return str(text[0])
        if text is not None:
            return str(text)
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)
