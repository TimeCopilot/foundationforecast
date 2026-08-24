from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

GIFT_EVAL_RESULTS_BASE = (
    "https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/results"
)

MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"

# GIFT-Eval leaderboard ranks on MASE + CRPS (WQL). Secondary metrics (MSE, etc.)
# can differ across library versions without indicating a failed replication.
REPLICATION_METRIC_COLS = [MASE_COL, CRPS_COL]

# Default verify tolerances. rtol=2.5% covers typical drift from uni2ts/torch/CUDA
# versions vs the original submission environment while still catching gross errors.
REPLICATION_ATOL = 1e-2
REPLICATION_RTOL = 2.5e-2


@lru_cache
def load_reference_results(
    model_slug: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    cache_root = cache_dir or Path(".pytest_cache") / "gift_eval" / "references"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"{model_slug}_all_results.csv"
    if not cache_file.exists():
        url = f"{GIFT_EVAL_RESULTS_BASE}/{model_slug}/all_results.csv"
        df = pd.read_csv(url)
        df.to_csv(cache_file, index=False)
    return pd.read_csv(cache_file)


def compare_results(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    atol: float = REPLICATION_ATOL,
    rtol: float = REPLICATION_RTOL,
) -> None:
    if actual.empty:
        raise AssertionError("Actual results are empty")
    if expected.empty:
        raise AssertionError("Expected results are empty")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True)[REPLICATION_METRIC_COLS],
        expected.reset_index(drop=True)[REPLICATION_METRIC_COLS],
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )
