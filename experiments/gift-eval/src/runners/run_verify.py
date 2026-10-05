from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.eval.jobs import (
    ci_output_root,
    load_ci_subset,
    load_replication_families,
    load_replication_matrix,
    load_replication_pilot_jobs,
    replication_output_root,
)
from src.eval.models import model_keys_with_reference
from src.verify.replication_table import write_replication_table
from src.verify.verify import ReplicationSkip, verify_all, verify_model

logging.basicConfig(level=logging.INFO)
app = typer.Typer()

DEFAULT_TABLE_PATH = Path("results/replication_table.csv")


def _resolve_model_keys(
    *,
    model_key: str | None,
    all_models: bool,
    ci: bool,
    replication: bool,
    run_id: str | None,
    replication_pilot: bool,
) -> tuple[list[str], Path, list | None]:
    if replication:
        if not run_id:
            raise typer.BadParameter("--replication requires --run-id RUN_ID")
        root = replication_output_root(run_id)
        if replication_pilot:
            jobs = load_replication_pilot_jobs()
        elif all_models or not model_key:
            jobs = load_replication_matrix()
        else:
            jobs = None
        if model_key:
            keys = [model_key]
        elif replication_pilot:
            keys = sorted({job.model_key for job in jobs or []})
        else:
            keys = list(load_replication_families())
        return keys, root, jobs

    if ci:
        jobs = load_ci_subset()
        return sorted({job.model_key for job in jobs}), ci_output_root(), jobs

    if model_key and all_models:
        raise typer.BadParameter("Use either --model-key or --all, not both")

    if model_key:
        return [model_key], Path("results"), None

    if all_models:
        return model_keys_with_reference(), Path("results"), None

    raise typer.BadParameter(
        "Specify --model-key KEY, --all, --ci, or --replication --run-id. "
        "Example: uv run python -m src.runners.run_verify --all"
    )


def _run_verify_models(
    model_keys: list[str],
    output_root: Path,
    *,
    require_complete: bool,
    registry: str = "default",
) -> None:
    passed: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for key in model_keys:
        try:
            verify_model(
                key,
                output_root,
                require_complete=require_complete,
                registry=registry,  # type: ignore[arg-type]
            )
            passed.append(key)
        except ReplicationSkip as exc:
            skipped.append((key, str(exc)))
            logging.warning("Skipped %s: %s", key, exc)
        except Exception as exc:
            failed.append((key, str(exc)))
            logging.error("Failed %s: %s", key, exc)

    logging.info(
        "Verify summary: passed=%s skipped=%s failed=%s",
        len(passed),
        len(skipped),
        len(failed),
    )
    if failed:
        details = "\n".join(f"  {key}: {error}" for key, error in failed)
        logging.error("Verification failed:\n%s", details)
        raise typer.Exit(code=1)


@app.command()
def main(
    model_key: Annotated[
        str | None,
        typer.Option(help="Verify one model against its HF reference CSV"),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Verify every model with a reference_slug"),
    ] = False,
    ci: Annotated[
        bool,
        typer.Option(help="Verify CI subset jobs (per-job layout under results/ci)"),
    ] = False,
    replication: Annotated[
        bool,
        typer.Option(help="Verify replication run under results/replication/RUN_ID"),
    ] = False,
    run_id: Annotated[
        str | None,
        typer.Option(help="Replication run id (required with --replication)"),
    ] = None,
    replication_pilot: Annotated[
        bool,
        typer.Option(
            help="With --replication, verify pilot jobs only (m4_weekly/short)"
        ),
    ] = False,
    output_root: Annotated[
        Path | None,
        typer.Option(help="Root directory containing benchmark outputs"),
    ] = None,
    table_output: Annotated[
        Path,
        typer.Option(help="Path to write the replication analysis CSV"),
    ] = DEFAULT_TABLE_PATH,
    verify_only: Annotated[
        bool,
        typer.Option(help="Build the replication table without strict verify"),
    ] = False,
    require_complete: Annotated[
        bool,
        typer.Option(
            help="Fail if any HF reference dataset is missing from actual results"
        ),
    ] = False,
) -> None:
    model_keys, default_root, replication_jobs = _resolve_model_keys(
        model_key=model_key,
        all_models=all_models,
        ci=ci,
        replication=replication,
        run_id=run_id,
        replication_pilot=replication_pilot,
    )
    resolved_output_root = output_root or default_root
    registry = "replication" if replication else "default"

    if ci and not verify_only:
        verify_all(load_ci_subset(), resolved_output_root)
    elif replication and replication_pilot and not verify_only:
        verify_all(
            replication_jobs or load_replication_pilot_jobs(),
            resolved_output_root,
            registry=registry,  # type: ignore[arg-type]
        )
    elif not verify_only:
        _run_verify_models(
            model_keys,
            resolved_output_root,
            require_complete=require_complete,
            registry=registry,
        )

    write_replication_table(
        model_keys,
        resolved_output_root,
        table_output,
        registry=registry,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
