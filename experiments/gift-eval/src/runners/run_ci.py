from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.eval.evaluate import run_gift_eval
from src.eval.jobs import ci_output_root, jobs_missing_timing, load_ci_subset
from src.verify.verify import verify_all

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


def _run_jobs_local(
    jobs: list,
    *,
    storage_path: Path,
    output_root: Path,
    overwrite_results: bool = True,
) -> None:
    for job in jobs:
        run_gift_eval(
            job,
            storage_path=storage_path,
            output_root=output_root,
            overwrite_results=overwrite_results,
        )


@app.command()
def main(
    local: Annotated[
        bool,
        typer.Option(help="Run jobs locally instead of dispatching to Modal"),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(help="Verify outputs against official HF reference CSVs"),
    ] = False,
    verify_only: Annotated[
        bool,
        typer.Option(help="Skip evaluation and only verify existing outputs"),
    ] = False,
    missing_timing_only: Annotated[
        bool,
        typer.Option(
            help="Rerun jobs that have results but no timing.json (local only)"
        ),
    ] = False,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Directory containing CI subset outputs"),
    ] = None,
    storage_path: Annotated[
        Path,
        typer.Option(help="Path to downloaded GIFT-Eval dataset"),
    ] = Path("data/gift-eval"),
) -> None:
    jobs = load_ci_subset()
    resolved_output_root = output_root or ci_output_root()
    if missing_timing_only:
        jobs = jobs_missing_timing(jobs, resolved_output_root)
        if not jobs:
            logging.info("All jobs already have timing.json")
            return
        logging.info("Rerunning %s jobs to backfill timing", len(jobs))
    if not verify_only:
        if local:
            _run_jobs_local(
                jobs,
                storage_path=storage_path,
                output_root=resolved_output_root,
            )
        else:
            raise typer.BadParameter(
                "Remote runs use Modal: "
                "`uv run modal run -m src.runners.run_modal::run_ci`"
            )

    if verify or verify_only:
        verify_all(jobs, resolved_output_root, storage_path=storage_path)


if __name__ == "__main__":
    app()
