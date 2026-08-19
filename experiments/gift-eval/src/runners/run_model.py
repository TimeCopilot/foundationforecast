from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.eval.evaluate import run_gift_eval
from src.eval.jobs import Job

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def main(
    model_key: Annotated[str, typer.Option(help="Model key from configs/models.yaml")],
    dataset_name: Annotated[str, typer.Option(help="GIFT-Eval dataset name")],
    term: Annotated[str, typer.Option(help="Forecast horizon term")],
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for benchmark outputs"),
    ] = Path("results"),
    storage_path: Annotated[
        Path,
        typer.Option(help="Path to downloaded GIFT-Eval dataset"),
    ] = Path("data/gift-eval"),
) -> None:
    job = Job(model_key=model_key, dataset_name=dataset_name, term=term)
    run_gift_eval(job, storage_path=storage_path, output_root=output_root)


if __name__ == "__main__":
    app()
