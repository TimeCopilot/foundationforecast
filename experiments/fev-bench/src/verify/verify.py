from __future__ import annotations

import json
import logging
from pathlib import Path

from numpy.testing import assert_allclose

from src.eval.jobs import Job, summary_json
from src.eval.models import reference_csv
from .reference import (
    REPLICATION_ATOL,
    REPLICATION_METRIC_COL,
    REPLICATION_RTOL,
    load_reference_results,
)

logger = logging.getLogger(__name__)


class ReplicationSkip(Exception):
    """Raised when a job has no public fev-bench reference to compare against."""


def verify_job(
    job: Job,
    output_root: Path,
    *,
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
) -> None:
    ref_name = reference_csv(job.model_key)
    if ref_name is None:
        raise ReplicationSkip(f"No reference_csv for model_key={job.model_key!r}")

    summary_path = summary_json(job, output_root)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with summary_path.open() as f:
        summary = json.load(f)

    actual_error = float(summary[REPLICATION_METRIC_COL])
    if not (actual_error == actual_error):  # NaN check
        raise AssertionError(f"NaN {REPLICATION_METRIC_COL} in {summary_path}")

    inference_time = summary.get("inference_time_s")
    if inference_time is None or float(inference_time) <= 0:
        raise AssertionError(f"inference_time_s must be > 0 in {summary_path}")

    expected_df = load_reference_results(ref_name)
    expected_row = expected_df.loc[expected_df["task_name"] == job.task_name]
    if expected_row.empty:
        raise AssertionError(
            f"No reference row for task_name={job.task_name!r} in {ref_name}"
        )
    expected_error = float(expected_row[REPLICATION_METRIC_COL].iloc[0])

    assert_allclose(
        actual_error,
        expected_error,
        atol=atol,
        rtol=rtol,
        err_msg=(
            f"test_error mismatch for {job.task_name}: "
            f"actual={actual_error} expected={expected_error}"
        ),
    )


def verify_all(
    jobs: list[Job],
    output_root: Path,
    *,
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
) -> None:
    for job in jobs:
        verify_job(job, output_root, atol=atol, rtol=rtol)
