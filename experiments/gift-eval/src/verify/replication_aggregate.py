"""Leaderboard-style aggregate replication checks (experiment grid only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import stats

from src.eval.jobs import Job, load_replication_matrix, result_csv
from src.eval.models import Registry, reference_slug
from src.verify.reference import CRPS_COL, MASE_COL, load_reference_results
from src.verify.verify import ReplicationSkip, load_actual_results

logger = logging.getLogger(__name__)

SEASONAL_NAIVE_REFERENCE_SLUG = "Seasonal_Naive"

# Geomean(MASE/SN, CRPS/SN) vs the HF submission on the same dataset set.
# Slightly looser than per-job verify is unnecessary at aggregate level; same
# rtol keeps the experiment aligned with CI semantics while tolerating outlier jobs.
REPLICATION_AGGREGATE_ATOL = 1e-3
REPLICATION_AGGREGATE_RTOL = 2.5e-2


@dataclass(frozen=True)
class AggregateScores:
    norm_mase: float
    norm_crps: float
    n_configs: int


@lru_cache
def seasonal_naive_by_dataset() -> pd.DataFrame:
    sn = load_reference_results(SEASONAL_NAIVE_REFERENCE_SLUG)
    return sn.rename(
        columns={
            MASE_COL: "sn_mase",
            CRPS_COL: "sn_crps",
        }
    )[["dataset", "sn_mase", "sn_crps"]]


def geomean_positive(series: pd.Series) -> float:
    values = series.astype(float)
    values = values[values > 0]
    if values.empty:
        return float("nan")
    return float(stats.gmean(values))


def leaderboard_aggregate(df: pd.DataFrame, sn: pd.DataFrame) -> AggregateScores:
    merged = df.merge(sn, on="dataset", how="inner")
    if merged.empty:
        return AggregateScores(float("nan"), float("nan"), 0)
    norm_mase = merged[MASE_COL] / merged["sn_mase"]
    norm_crps = merged[CRPS_COL] / merged["sn_crps"]
    return AggregateScores(
        geomean_positive(norm_mase),
        geomean_positive(norm_crps),
        len(merged),
    )


def missing_matrix_jobs(model_key: str, output_root) -> list[Job]:
    from pathlib import Path

    root = Path(output_root)
    missing: list[Job] = []
    for job in load_replication_matrix():
        if job.model_key != model_key:
            continue
        if not result_csv(job, root).exists():
            missing.append(job)
    return missing


def verify_replication_aggregate(
    model_key: str,
    output_root,
    *,
    require_complete: bool = False,
    registry: Registry = "replication",
    atol: float = REPLICATION_AGGREGATE_ATOL,
    rtol: float = REPLICATION_AGGREGATE_RTOL,
) -> AggregateScores:
    from pathlib import Path

    root = Path(output_root)
    slug = reference_slug(model_key, registry=registry)
    if slug is None:
        raise ReplicationSkip(f"No reference slug for model_key={model_key!r}")

    if require_complete:
        missing = missing_matrix_jobs(model_key, root)
        if missing:
            sample = ", ".join(f"{j.dataset_name}/{j.term}" for j in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise AssertionError(
                f"{model_key}: missing {len(missing)} replication matrix jobs "
                f"(e.g. {sample}{suffix})"
            )

    actual = load_actual_results(model_key, root)
    expected = load_reference_results(slug)
    datasets = sorted(set(actual["dataset"]) & set(expected["dataset"]))
    if not datasets:
        raise AssertionError(f"{model_key}: no overlapping datasets with HF reference")

    sn = seasonal_naive_by_dataset()
    actual_sub = actual[actual["dataset"].isin(datasets)]
    expected_sub = expected[expected["dataset"].isin(datasets)]

    rep = leaderboard_aggregate(actual_sub, sn)
    hf = leaderboard_aggregate(expected_sub, sn)
    if rep.n_configs != hf.n_configs or rep.n_configs != len(datasets):
        raise AssertionError(
            f"{model_key}: aggregate config count mismatch "
            f"(rep={rep.n_configs}, hf={hf.n_configs}, overlap={len(datasets)})"
        )

    for label, rep_val, hf_val in (
        ("geomean(MASE/Seasonal_Naive)", rep.norm_mase, hf.norm_mase),
        ("geomean(CRPS/Seasonal_Naive)", rep.norm_crps, hf.norm_crps),
    ):
        try:
            assert_allclose(
                np.array([rep_val], dtype=float),
                np.array([hf_val], dtype=float),
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as exc:
            delta_pct = 100 * (rep_val - hf_val) / hf_val if hf_val else float("nan")
            raise AssertionError(
                f"{model_key}: {label} differs from HF submission aggregate "
                f"(rep={rep_val:.6g}, hf={hf_val:.6g}, delta={delta_pct:+.2f}%, "
                f"n={rep.n_configs}, atol={atol}, rtol={rtol})"
            ) from exc

    logger.info(
        "%s: aggregate OK on %s configs — rep MASE/SN=%.4f CRPS/SN=%.4f "
        "(HF %.4f / %.4f)",
        model_key,
        rep.n_configs,
        rep.norm_mase,
        rep.norm_crps,
        hf.norm_mase,
        hf.norm_crps,
    )
    return rep


def verify_replication_aggregates(
    model_keys: list[str],
    output_root,
    *,
    require_complete: bool = False,
    registry: Registry = "replication",
    atol: float = REPLICATION_AGGREGATE_ATOL,
    rtol: float = REPLICATION_AGGREGATE_RTOL,
) -> None:
    passed: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for key in model_keys:
        try:
            verify_replication_aggregate(
                key,
                output_root,
                require_complete=require_complete,
                registry=registry,
                atol=atol,
                rtol=rtol,
            )
            passed.append(key)
        except ReplicationSkip as exc:
            skipped.append((key, str(exc)))
            logger.warning("Skipped %s: %s", key, exc)
        except Exception as exc:
            failed.append((key, str(exc)))
            logger.error("Failed %s: %s", key, exc)

    logger.info(
        "Aggregate verify summary: passed=%s skipped=%s failed=%s",
        len(passed),
        len(skipped),
        len(failed),
    )
    if failed:
        details = "\n".join(f"  {key}: {error}" for key, error in failed)
        raise AssertionError(f"Aggregate replication verification failed:\n{details}")
