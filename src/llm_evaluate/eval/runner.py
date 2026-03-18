from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from llm_evaluate.config import RunConfig
from llm_evaluate.data.loaders import DatasetLoader
from llm_evaluate.metrics.manager import MetricsManager
from llm_evaluate.models.factory import build_model_adapter


class EvaluationRunner:
    def __init__(self, config: RunConfig):
        self.config = config

    def run(self) -> Path:
        run_id = f"{self.config.run_name}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path(self.config.output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        loader = DatasetLoader()
        samples = loader.load(self.config.dataset)
        model = build_model_adapter(self.config.model)
        metrics_manager = MetricsManager(self.config.metrics.names)

        predictions = []
        sample_scores = []
        for sample in samples:
            pred = model.generate(prompt=sample.prompt, image=sample.image)
            scores = metrics_manager.score_sample(pred, sample.answer)
            sample_scores.append(scores)
            row = asdict(sample) | {"prediction": pred, "scores": scores}
            predictions.append(row)

        aggregate_metrics = metrics_manager.aggregate(sample_scores)

        (out_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions),
            encoding="utf-8",
        )
        (out_dir / "metrics.json").write_text(
            json.dumps(aggregate_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "run_config.snapshot.yaml").write_text(
            _dump_yaml_with_fallback(self.config.model_dump()),
            encoding="utf-8",
        )
        return out_dir


def _dump_yaml_with_fallback(payload: dict) -> str:
    try:
        import yaml

        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except ModuleNotFoundError:
        return json.dumps(payload, ensure_ascii=False, indent=2)
