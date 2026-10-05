"""Shared test utilities.

Use generate_series for panel data; manual DataFrame only for controlled y
patterns or edge-case lengths.
"""

import numpy as np
import pandas as pd
from utilsforecast.data import generate_series as _generate_series
from utilsforecast.processing import make_future_dataframe

from foundationforecast.core.forecaster import ExogCapableForecaster, QuantileConverter
from foundationforecast.core.quantiles import quantile_column_name


def generate_series(n_series, freq, **kwargs):
    df = _generate_series(n_series, freq, **kwargs)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


def generate_panel_with_futr_exog(
    n_series: int,
    freq: str,
    h: int,
    *,
    min_length: int = 32,
    max_length: int = 32,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Panel with known-future dynamic exog columns and matching ``X_df``."""
    df = generate_series(
        n_series,
        freq=freq,
        min_length=min_length,
        max_length=max_length,
        equal_ends=True,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = df.groupby("unique_id", observed=True).cumcount().astype(float)
    futr_exog_list = ["x1", "x2"]

    last_times = df.groupby("unique_id", observed=True)["ds"].max()
    X_df = make_future_dataframe(
        uids=last_times.index.tolist(),
        last_times=last_times,
        h=h,
        freq=freq,
    )
    rng_hor = np.random.default_rng(seed + 1)
    X_df["x1"] = rng_hor.normal(size=len(X_df))
    X_df["x2"] = (
        X_df.groupby("unique_id", observed=True).cumcount().astype(float) + 100.0
    )
    return df, X_df, futr_exog_list


def panel_with_futr_exog_horizon(
    df_hist: pd.DataFrame,
    X_df: pd.DataFrame,
) -> pd.DataFrame:
    """History plus horizon rows (exog + placeholder ``y``) for cross-validation."""
    future = X_df.copy()
    last_y = df_hist.groupby("unique_id", observed=True)["y"].last()
    future["y"] = future["unique_id"].map(last_y)
    return pd.concat([df_hist, future], ignore_index=True).sort_values(
        ["unique_id", "ds"]
    )


def generate_series_with_anomalies(
    n_series: int = 2,
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 50,
    anomaly_positions: list[int] | None = None,
    anomaly_magnitude: float = 5.0,
) -> pd.DataFrame:
    """Generate time series with artificial anomalies for testing."""
    df = generate_series(
        n_series=n_series,
        freq=freq,
        min_length=min_length,
        max_length=max_length,
    )

    if anomaly_positions is not None:
        for series_id in df["unique_id"].unique():
            series_data = df[df["unique_id"] == series_id].copy()
            for pos in anomaly_positions:
                if pos < len(series_data):
                    anomaly_idx = series_data.index[pos]
                    df.loc[anomaly_idx, "y"] += anomaly_magnitude

    return df


class DummyModel(ExogCapableForecaster):
    def __init__(self, alias: str = "dummy", reuse_loaded_model: bool = True):
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        self.alias = alias

    def _forecast_univariate(
        self, df, h, freq=None, level=None, quantiles=None, panel=None
    ):
        _ = panel
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        last_times = df.groupby("unique_id")["ds"].max()
        uids = last_times.index.tolist()
        fcst = make_future_dataframe(
            uids=uids,
            last_times=last_times,
            h=h,
            freq=freq,
        )
        fcst[self.alias] = 1.0
        if qc.quantiles:
            for q in qc.quantiles:
                fcst[quantile_column_name(self.alias, q)] = fcst[self.alias] + q
            fcst = qc.maybe_convert_quantiles_to_level(fcst, models=[self.alias])
        return fcst


class SeasonalNaiveModel(ExogCapableForecaster):
    alias = "SeasonalNaive"

    def __init__(
        self,
        season_length: int | None = None,
        reuse_loaded_model: bool = True,
    ):
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        self.season_length = season_length

    def _forecast_univariate(
        self, df, h, freq=None, level=None, quantiles=None, panel=None
    ):
        _ = panel
        freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(freq)
        results = []
        for uid, group in df.groupby("unique_id"):
            y = group["y"].values
            effective_length = min(season_length, len(y))
            last_times = group["ds"].max()
            future = make_future_dataframe(
                uids=[uid],
                last_times=pd.Series([last_times], index=[uid]),
                h=h,
                freq=freq,
            )
            seasonal_values = y[-effective_length:]
            future[self.alias] = [
                seasonal_values[i % effective_length] for i in range(h)
            ]
            results.append(future)
        return pd.concat(results, ignore_index=True)
