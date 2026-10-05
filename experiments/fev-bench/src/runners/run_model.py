from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.eval.evaluate import run_fev_eval
from src.eval.jobs import Job

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def main(
    model_key: Annotated[str, typer.Option(help="Model key from configs/models.yaml")],
    task_name: Annotated[str, typer.Option(help="fev-bench task_name (known-only)")],
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for benchmark outputs"),
    ] = Path("results"),
    num_proc: Annotated[
        int,
        typer.Option(help="fev dataset preprocessing processes per window"),
    ] = 8,
) -> None:
    job = Job(model_key=model_key, task_name=task_name)
    run_fev_eval(job, output_root=output_root, num_proc=num_proc)


if __name__ == "__main__":
    app()
