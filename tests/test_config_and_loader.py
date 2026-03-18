from pathlib import Path

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
