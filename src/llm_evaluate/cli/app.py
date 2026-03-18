from __future__ import annotations

from pathlib import Path

import typer

from llm_evaluate.config import load_run_config
from llm_evaluate.eval.runner import EvaluationRunner

app = typer.Typer(help="LLM/VLM evaluation framework")


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, readable=True, help="YAML config path"),
) -> None:
    run_config = load_run_config(config)
    runner = EvaluationRunner(run_config)
    out_dir = runner.run()
    typer.echo(f"Evaluation completed. Output: {out_dir}")


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
    run_config = load_run_config(config)
    typer.echo(f"Config valid: run_name={run_config.run_name}, backend={run_config.model.backend}")


if __name__ == "__main__":
    app()
