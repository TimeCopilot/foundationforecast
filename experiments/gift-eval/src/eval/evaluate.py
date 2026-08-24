from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from timecopilot_gift_eval import GIFTEval, GluonTSPredictor

from .jobs import Job, job_output_dir, result_csv, timing_json
from .models import build_model

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096
DEFAULT_PREDICTOR_BATCH_SIZE = 1024
DEFAULT_EVAL_BATCH_SIZE = 512


def run_gift_eval(
    job: Job,
    *,
    storage_path: Path | str,
    output_root: Path | str = Path("results"),
    overwrite_results: bool = False,
) -> Path:
    output_path = job_output_dir(job, Path(output_root))
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running GIFT-Eval job model=%s dataset=%s term=%s",
        job.model_key,
        job.dataset_name,
        job.term,
    )

    predictor = GluonTSPredictor(
        forecaster=build_model(job.model_key),
        max_length=DEFAULT_MAX_LENGTH,
        batch_size=DEFAULT_PREDICTOR_BATCH_SIZE,
    )
    gifteval = GIFTEval(
        dataset_name=job.dataset_name,
        term=job.term,
        output_path=output_path,
        storage_path=storage_path,
    )
    started_at = time.perf_counter()
    gifteval.evaluate_predictor(
        predictor,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        overwrite_results=overwrite_results,
    )
    elapsed_seconds = time.perf_counter() - started_at

    timing_path = timing_json(job, Path(output_root))
    timing_path.write_text(
        json.dumps(
            {
                "model_key": job.model_key,
                "dataset_name": job.dataset_name,
                "term": job.term,
                "elapsed_seconds": elapsed_seconds,
            },
            indent=2,
        )
    )

    csv_path = result_csv(job, Path(output_root))
    logger.info("Wrote results to %s (%.1fs)", csv_path, elapsed_seconds)
    return csv_path
