"""Shared test utilities.

Use generate_series for panel data; manual DataFrame only for controlled y
patterns or edge-case lengths.
"""

import pandas as pd
from utilsforecast.data import generate_series as _generate_series
from utilsforecast.processing import make_future_dataframe

from foundationforecast.core.forecaster import Forecaster, QuantileConverter


def generate_series(n_series, freq, **kwargs):
    df = _generate_series(n_series, freq, **kwargs)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


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


class DummyModel(Forecaster):
    def __init__(self, alias: str = "dummy"):
        self.alias = alias

    def forecast(self, df, h, freq=None, level=None, quantiles=None, panel=None):
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
                fcst[f"{self.alias}-q-{int(q * 100)}"] = fcst[self.alias] + q
            fcst = qc.maybe_convert_quantiles_to_level(fcst, models=[self.alias])
        return fcst


class SeasonalNaiveModel(Forecaster):
    alias = "SeasonalNaive"

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def forecast(self, df, h, freq=None, level=None, quantiles=None, panel=None):
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
