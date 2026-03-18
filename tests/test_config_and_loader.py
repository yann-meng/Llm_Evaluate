from pathlib import Path

import pandas as pd

from llm_evaluate.config import load_run_config
from llm_evaluate.data.loaders import DatasetLoader


def test_load_run_config() -> None:
    cfg = load_run_config("configs/demo_local.yaml")
    assert cfg.datasets[0].source == "local"
    assert cfg.model.backend == "openai_compatible"


def test_local_loader() -> None:
    loader = DatasetLoader()
    cfg = load_run_config("configs/demo_local.yaml")
    samples = loader.load(cfg.datasets[0])
    assert len(samples) == 2
    assert samples[0].prompt


def test_csv_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("id,prompt,answer\n1,hello,world\n", encoding="utf-8")
    cfg = load_run_config("configs/demo_local.yaml")
    cfg.datasets[0].path = str(csv_path)
    samples = DatasetLoader().load(cfg.datasets[0])
    assert samples[0].answer == "world"


def test_multi_dataset_config() -> None:
    cfg = load_run_config("configs/demo_multi_dataset.yaml")
    assert len(cfg.datasets) == 2
    assert {ds.dataset_id for ds in cfg.datasets} == {"math_zh", "qa_en"}



def test_local_transformers_qwen3_config() -> None:
    cfg = load_run_config("configs/demo_local_transformers_qwen3_squad.yaml")
    assert cfg.model.backend == "transformers"
    assert cfg.model.device_map == "auto"
    assert cfg.datasets[0].name == "squad"


def test_parquet_loader(tmp_path: Path) -> None:
    parquet_path = tmp_path / "eval.parquet"
    pd.DataFrame(
        [
            {"id": "1", "prompt": "hello", "answer": "world"},
            {"id": "2", "prompt": "foo", "answer": "bar"},
        ]
    ).to_parquet(parquet_path)
    cfg = load_run_config("configs/demo_local.yaml")
    cfg.datasets[0].path = str(parquet_path)
    samples = DatasetLoader().load(cfg.datasets[0])
    assert len(samples) == 2
    assert samples[1].answer == "bar"


def test_parquet_dir_loader_by_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"id": "11", "prompt": "q1", "answer": "a1"},
            {"id": "12", "prompt": "q2", "answer": "a2"},
        ]
    ).to_parquet(data_dir / "test-00000-of-00001.parquet")
    cfg = load_run_config("configs/demo_local.yaml")
    cfg.datasets[0].path = str(data_dir)
    cfg.datasets[0].split = "test"
    samples = DatasetLoader().load(cfg.datasets[0])
    assert len(samples) == 2
    assert samples[0].prompt == "q1"
