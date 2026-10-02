from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.eval.evaluate import run_fev_eval
from src.eval.jobs import ci_output_root, load_ci_subset
from src.verify.verify import verify_all

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def main(
    local: Annotated[
        bool,
        typer.Option(help="Run jobs locally instead of dispatching to Modal"),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(help="Verify outputs against fev-bench reference CSVs"),
    ] = False,
    verify_only: Annotated[
        bool,
        typer.Option(help="Skip evaluation and only verify existing outputs"),
    ] = False,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Directory containing CI subset outputs"),
    ] = None,
    num_proc: Annotated[int, typer.Option()] = 8,
) -> None:
    jobs = load_ci_subset()
    resolved_output_root = output_root or ci_output_root()

    if not verify_only:
        if local:
            for job in jobs:
                run_fev_eval(job, output_root=resolved_output_root, num_proc=num_proc)
        else:
            raise typer.BadParameter(
                "Remote runs use Modal: "
                "`uv run modal run -m src.runners.run_modal::run_ci`"
            )

    if verify or verify_only:
        verify_all(jobs, resolved_output_root)


if __name__ == "__main__":
    app()
