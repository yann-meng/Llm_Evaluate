from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from llm_evaluate.config import DatasetConfig, RunConfig
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
        model = build_model_adapter(self.config.model)
        metrics_manager = MetricsManager(self.config.metrics.names)
        summary: dict[str, dict[str, float]] = {}
        for dataset_config in self._iter_datasets():
            dataset_dir = out_dir / dataset_config.dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            samples = loader.load(dataset_config)

            predictions = []
            sample_scores = []
            for sample in samples:
                pred = model.generate(prompt=sample.prompt, image=sample.image)
                scores = metrics_manager.score_sample(pred, sample.answer)
                sample_scores.append(scores)
                row = asdict(sample) | {"prediction": pred, "scores": scores}
                predictions.append(row)

            aggregate_metrics = metrics_manager.aggregate(sample_scores)
            summary[dataset_config.dataset_id] = aggregate_metrics
            (dataset_dir / "predictions.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions),
                encoding="utf-8",
            )
            (dataset_dir / "metrics.json").write_text(
                json.dumps(aggregate_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        (out_dir / "summary_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "summary_metrics.html").write_text(
            _build_summary_html(summary), encoding="utf-8"
        )
        (out_dir / "run_config.snapshot.yaml").write_text(
            _dump_yaml_with_fallback(self.config.model_dump()),
            encoding="utf-8",
        )
        return out_dir

    def _iter_datasets(self) -> list[DatasetConfig]:
        if self.config.datasets:
            return self.config.datasets
        if self.config.dataset:
            return [self.config.dataset]
        raise ValueError("No dataset configured.")


def _dump_yaml_with_fallback(payload: dict) -> str:
    try:
        import yaml

        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except ModuleNotFoundError:
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_summary_html(summary: dict[str, dict[str, float]]) -> str:
    metric_names = sorted({metric for ds in summary.values() for metric in ds})
    headers = "".join(f"<th>{name}</th>" for name in metric_names)
    rows = []
    for dataset_id, metrics in summary.items():
        cells = "".join(f"<td>{metrics.get(name, float('nan')):.4f}</td>" for name in metric_names)
        rows.append(f"<tr><td>{dataset_id}</td>{cells}</tr>")
    row_html = "\n".join(rows)
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>Evaluation Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h2>Dataset Metrics Comparison</h2>
  <table>
    <thead><tr><th>dataset_id</th>{headers}</tr></thead>
    <tbody>{row_html}</tbody>
  </table>
</body>
</html>"""
