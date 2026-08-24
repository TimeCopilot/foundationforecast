from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from timecopilot_gift_eval import GIFTEval

from src.eval.jobs import Job, result_csv
from src.eval.models import load_models_config, reference_slug
from .reference import (
    REPLICATION_ATOL,
    REPLICATION_RTOL,
    compare_results,
    load_reference_results,
)

logger = logging.getLogger(__name__)


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
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
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
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
) -> None:
    for job in jobs:
        verify_job(
            job,
            output_root,
            storage_path=storage_path,
            atol=atol,
            rtol=rtol,
        )


def load_actual_results(model_key: str, output_root: Path) -> pd.DataFrame:
    consolidated = output_root / model_key / "all_results.csv"
    if consolidated.exists():
        return pd.read_csv(consolidated)

    job_csvs = sorted((output_root / model_key).glob("**/all_results.csv"))
    if not job_csvs:
        raise FileNotFoundError(
            f"No results found for {model_key!r} under {output_root}"
        )

    return (
        pd.concat([pd.read_csv(path) for path in job_csvs], ignore_index=True)
        .drop_duplicates(subset=["dataset"])
        .reset_index(drop=True)
    )


def verify_model(
    model_key: str,
    output_root: Path,
    *,
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
    require_complete: bool = False,
) -> None:
    slug = reference_slug(model_key)
    if slug is None:
        raise ReplicationSkip(f"No reference slug for model_key={model_key!r}")

    actual = load_actual_results(model_key, output_root)
    if actual.isna().any().any():
        raise AssertionError(f"NaN values found in actual results for {model_key!r}")

    expected = load_reference_results(slug)
    common = sorted(set(actual["dataset"]) & set(expected["dataset"]))
    missing = sorted(set(expected["dataset"]) - set(actual["dataset"]))

    if missing:
        message = (
            f"{model_key}: missing {len(missing)}/{len(expected)} "
            f"HF datasets (have {len(actual)}, need overlap with reference)"
        )
        if require_complete:
            raise AssertionError(message)
        logger.warning(message)

    if not common:
        raise AssertionError(f"{model_key}: no overlapping datasets with HF reference")

    actual_sub = actual[actual["dataset"].isin(common)].sort_values("dataset")
    expected_sub = expected[expected["dataset"].isin(common)].sort_values("dataset")
    compare_results(actual_sub, expected_sub, atol=atol, rtol=rtol)
    logger.info(
        "%s: verified %s/%s datasets against HF reference %s",
        model_key,
        len(common),
        len(expected),
        slug,
    )


def model_keys_with_reference() -> list[str]:
    models = load_models_config()
    return [
        model_key
        for model_key, spec in models.items()
        if spec.get("reference_slug") is not None
    ]
