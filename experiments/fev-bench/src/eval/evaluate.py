from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .forecast import forecast_task
from .jobs import Job, job_output_dir, load_task, summary_csv, summary_json, timing_json
from .models import build_model

logger = logging.getLogger(__name__)


def run_fev_eval(
    job: Job,
    *,
    output_root: Path | str = Path("results"),
    num_proc: int = 8,
) -> Path:
    output_path = job_output_dir(job, Path(output_root))
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running fev-bench job model=%s task=%s",
        job.model_key,
        job.task_name,
    )

    task = load_task(job.task_name)
    forecaster = build_model(job.model_key)
    model_display = getattr(forecaster, "alias", job.model_key)

    started_at = time.perf_counter()
    predictions_per_window = forecast_task(
        task,
        forecaster,
        num_proc=num_proc,
    )
    elapsed_seconds = time.perf_counter() - started_at

    summary = task.evaluation_summary(
        predictions_per_window,
        model_name=model_display,
        inference_time_s=elapsed_seconds,
    )

    summary_path = summary_json(job, Path(output_root))
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    csv_path = summary_csv(job, Path(output_root))
    pd.DataFrame([summary]).to_csv(csv_path, index=False)

    timing_path = timing_json(job, Path(output_root))
    timing_path.write_text(
        json.dumps(
            {
                "model_key": job.model_key,
                "task_name": job.task_name,
                "elapsed_seconds": elapsed_seconds,
            },
            indent=2,
        )
    )

    logger.info(
        "Wrote %s test_error=%s (%.1fs)",
        csv_path,
        summary.get("test_error"),
        elapsed_seconds,
    )
    return csv_path
