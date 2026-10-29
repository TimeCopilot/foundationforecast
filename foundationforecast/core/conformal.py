from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp

if TYPE_CHECKING:
    from .forecaster import ExogCapableForecaster


@dataclass
class ConformalIntervals:
    """Cross-validation settings for distribution-free prediction intervals.

    Args:
        n_windows: Number of rolling backtest windows used to estimate residuals.
            Must be at least 2.
        h: Forecast horizon used when estimating residuals. Set automatically
            from ``forecast(h=...)`` on each call.
        method: ``conformal_distribution`` builds bounds from mean ± error
            paths; ``conformal_error`` uses quantiles of absolute errors per
            horizon step.
    """

    n_windows: int = 2
    h: int = 1
    method: str = "conformal_distribution"

    def __post_init__(self) -> None:
        if self.n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed = ("conformal_distribution", "conformal_error")
        if self.method not in allowed:
            raise ValueError(f"method must be one of {allowed}")


def validate_conformal_samples(n_samples: int, h: int, n_windows: int) -> None:
    min_samples = 2 * h + 1
    if n_samples < min_samples:
        raise ValueError(
            f"Prediction intervals require at least {min_samples:,} samples; "
            f"series has {n_samples:,}."
        )
    required = n_windows * h + 1
    if n_samples < required:
        raise ValueError(
            f"Prediction intervals require at least {required:,} samples for "
            f"n_windows={n_windows} and h={h}; series has {n_samples:,}."
        )


def effective_n_windows(n_samples: int, h: int, n_windows: int) -> int:
    capped = min(n_windows, (n_samples - 1) // h)
    if capped < 2:
        raise ValueError(
            f"Prediction intervals require at least {2 * h + 1:,} samples; "
            f"series has {n_samples:,}."
        )
    return capped


def add_conformal_distribution_intervals(
    mean: np.ndarray,
    cs: np.ndarray,
    level: list[int | float],
) -> dict[str, np.ndarray]:
    level_sorted = sorted(level)
    alphas = [100 - lv for lv in level_sorted]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    mean_row = mean.reshape(1, -1)
    scores = np.vstack([mean_row - cs, mean_row + cs])
    quantiles = np.quantile(scores, cuts, axis=0)
    quantiles = quantiles.reshape(len(cuts), -1)
    lo_cols = [f"lo-{lv}" for lv in reversed(level_sorted)]
    hi_cols = [f"hi-{lv}" for lv in level_sorted]
    out: dict[str, np.ndarray] = {}
    for i, col in enumerate(lo_cols + hi_cols):
        out[col] = quantiles[i]
    return out


def add_conformal_error_intervals(
    mean: np.ndarray,
    cs: np.ndarray,
    level: list[int | float],
) -> dict[str, np.ndarray]:
    margin = {lv: np.quantile(cs, lv / 100, axis=0) for lv in level}
    out: dict[str, np.ndarray] = {}
    for lv in reversed(sorted(level)):
        out[f"lo-{lv}"] = mean - margin[lv]
    for lv in sorted(level):
        out[f"hi-{lv}"] = mean + margin[lv]
    return out


def _apply_conformal_method(
    mean: np.ndarray,
    cs: np.ndarray,
    level: list[int | float],
    method: str,
) -> dict[str, np.ndarray]:
    if method == "conformal_distribution":
        return add_conformal_distribution_intervals(mean, cs, level)
    if method == "conformal_error":
        return add_conformal_error_intervals(mean, cs, level)
    raise ValueError(f"Unsupported conformal method: {method}")


def add_conformal_columns_to_forecast(
    fcst_df: pd.DataFrame,
    alias: str,
    scores_by_id: dict[object, np.ndarray],
    level: list[int | float],
    method: str,
) -> pd.DataFrame:
    fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
    for uid, cs in scores_by_id.items():
        mask = fcst_df["unique_id"] == uid
        if not mask.any():
            continue
        sub = fcst_df.loc[mask].sort_values("ds")
        mean = sub[alias].to_numpy()
        bands = _apply_conformal_method(mean, cs, level, method)
        for col_suffix, values in bands.items():
            side, lv = col_suffix.split("-", 1)
            out_col = f"{alias}-{side}-{lv}"
            fcst_df.loc[sub.index, out_col] = values
    return fcst_df


def conformity_scores_abs_errors(
    y: np.ndarray,
    h: int,
    n_windows: int,
    forecast_values: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    n_samples = y.size
    n_windows_eff = effective_n_windows(n_samples, h, n_windows)
    test_size = n_windows_eff * h
    cs = np.empty((n_windows_eff, h), dtype=float)
    for i_window in range(n_windows_eff):
        train_end = n_samples - test_size + i_window * h
        y_test = y[train_end : train_end + h]
        fcst = forecast_values(train_end, i_window)
        cs[i_window] = np.abs(fcst - y_test)
    return cs


def compute_panel_conformity_scores(
    forecaster: ExogCapableForecaster,
    df: pd.DataFrame,
    h: int,
    freq: str,
    prediction_intervals: ConformalIntervals,
    *,
    exog_cols: list[str],
) -> dict[object, np.ndarray]:
    """Absolute CV errors of the point forecast, keyed by ``unique_id``.

    Each window is forecast from that series' training slice only, so callers
    must not forward a precomputed panel or the forecast horizon ``X_df``.
    """
    from .exog.covariates import exog_strategy_disabled, sort_exog_panel

    scores: dict[object, np.ndarray] = {}
    work_df = df
    if exog_cols and not exog_strategy_disabled(forecaster.exog_strategy):
        work_df = sort_exog_panel(df)

    for uid, group in work_df.groupby("unique_id", sort=False):
        series = group.sort_values("ds")
        y = series["y"].to_numpy(dtype=float)
        validate_conformal_samples(len(y), h, prediction_intervals.n_windows)

        def forecast_values(
            train_end: int,
            _i_window: int,
            _series: pd.DataFrame = series,
        ) -> np.ndarray:
            train = _series.iloc[:train_end]
            valid = _series.iloc[train_end : train_end + h]
            horizon_x = valid[["unique_id", "ds", *exog_cols]] if exog_cols else None
            pred = forecaster._forecast_with_exog(
                train,
                h,
                freq,
                None,
                None,
                None,
                horizon_x,
                None,
                exog_cols or None,
                lambda: forecaster._forecast_univariate(
                    train, h, freq, None, None, None
                ),
            )
            pred = pred.sort_values("ds")
            return pred[forecaster.alias].to_numpy(dtype=float)

        scores[uid] = conformity_scores_abs_errors(
            y,
            h,
            prediction_intervals.n_windows,
            forecast_values,
        )
    return scores
