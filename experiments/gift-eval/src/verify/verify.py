from __future__ import annotations

from pathlib import Path

import pandas as pd
from timecopilot_gift_eval import GIFTEval

from src.eval.jobs import Job, result_csv
from src.eval.models import reference_slug
from .reference import compare_results, load_reference_results


class ReplicationSkip(Exception):
    """Raised when a job has no public HF reference to compare against."""


def ds_config(job: Job, storage_path: Path | str) -> str:
    gifteval = GIFTEval(
        dataset_name=job.dataset_name,
        term=job.term,
        storage_path=storage_path,
    )
    return gifteval.ds_config


def verify_job(
    job: Job,
    output_root: Path,
    *,
    storage_path: Path | str | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    slug = reference_slug(job.model_key)
    if slug is None:
        raise ReplicationSkip(f"No reference slug for model_key={job.model_key!r}")

    csv_path = result_csv(job, output_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    actual_df = pd.read_csv(csv_path)
    if actual_df.isna().any().any():
        raise AssertionError(f"NaN values found in actual results at {csv_path}")

    expected_df = load_reference_results(slug)
    dataset_key = (
        ds_config(job, storage_path)
        if storage_path is not None
        else actual_df["dataset"].iloc[0]
    )

    expected_row = expected_df.loc[expected_df["dataset"] == dataset_key]
    actual_row = actual_df.loc[actual_df["dataset"] == dataset_key]
    if actual_row.empty:
        actual_row = actual_df

    compare_results(actual_row, expected_row, atol=atol, rtol=rtol)


def verify_all(
    jobs: list[Job],
    output_root: Path,
    *,
    storage_path: Path | str | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    for job in jobs:
        verify_job(
            job,
            output_root,
            storage_path=storage_path,
            atol=atol,
            rtol=rtol,
        )
