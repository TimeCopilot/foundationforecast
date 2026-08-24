from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

GIFT_EVAL_RESULTS_BASE = (
    "https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/results"
)

TARGET_COLS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]

MASE_COL = "eval_metrics/MASE[0.5]"
CRPS_COL = "eval_metrics/mean_weighted_sum_quantile_loss"


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
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    if actual.empty:
        raise AssertionError("Actual results are empty")
    if expected.empty:
        raise AssertionError("Expected results are empty")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True)[TARGET_COLS],
        expected.reset_index(drop=True)[TARGET_COLS],
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )
