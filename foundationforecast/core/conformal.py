from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp


@dataclass
class ConformalIntervals:
    """Cross-validation settings for distribution-free prediction intervals.

    Args:
        n_windows: Number of rolling backtest windows used to estimate residuals.
            Must be at least 2.
        method: ``conformal_distribution`` builds bounds from mean ± error
            paths; ``conformal_error`` uses quantiles of absolute errors per
            horizon step.
        step_size: Gap between consecutive calibration cutoffs. ``None``
            (default) uses the forecast horizon (non-overlapping windows).
    """

    n_windows: int = 2
    method: str = "conformal_distribution"
    step_size: int | None = None

    def __post_init__(self) -> None:
        if self.n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed = ("conformal_distribution", "conformal_error")
        if self.method not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        if self.step_size is not None and self.step_size < 1:
            raise ValueError("step_size must be a positive integer")


def validate_conformal_samples(
    n_samples: int, h: int, n_windows: int, step_size: int
) -> None:
    min_samples = h + (n_windows - 1) * step_size + 1
    if n_samples < min_samples:
        raise ValueError(
            "Minimum required samples in each series for the prediction "
            f"intervals settings are: {min_samples:,}, shortest series has: "
            f"{n_samples:,}."
        )


def add_conformal_distribution_intervals(
    fcst_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    model_names: list[str],
    level: list[int | float],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> pd.DataFrame:
    """Add intervals from quantiles of mean ± error paths."""
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    for model in model_names:
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
        scores = np.vstack([mean - scores, mean + scores])
        quantiles = np.quantile(scores, cuts, axis=0)
        quantiles = quantiles.reshape(len(cuts), -1).T
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles)
    return fcst_df


def add_conformal_error_intervals(
    fcst_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    model_names: list[str],
    level: list[int | float],
    cs_n_windows: int,
    cs_h: int,
    n_series: int,
    horizon: int,
) -> pd.DataFrame:
    """Add intervals from quantiles of absolute errors."""
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    cuts = [lv / 100 for lv in level]
    for model in model_names:
        mean = fcst_df[model].to_numpy().ravel()
        scores = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
        scores = scores[:, :, :horizon]
        quantiles = np.quantile(scores, cuts, axis=0)
        quantiles = quantiles.reshape(len(cuts), -1)
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        quantiles = np.vstack([mean - quantiles[::-1], mean + quantiles]).T
        fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, quantiles)
    return fcst_df


def get_conformal_method(method: str) -> Callable[..., pd.DataFrame]:
    available = {
        "conformal_distribution": add_conformal_distribution_intervals,
        "conformal_error": add_conformal_error_intervals,
    }
    if method not in available:
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available)}"
        )
    return available[method]
