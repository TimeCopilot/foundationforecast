"""Lightweight forecasters used to benchmark the FoundationForecast pipeline.

These models are deliberately trivial: the goal is to measure the cost of the
library's own data plumbing (validation, splitting, merging, quantile handling)
without downloading or running any foundation model weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from utilsforecast.processing import make_future_dataframe

from foundationforecast.core.forecaster import Forecaster, QuantileConverter


class ConstantModel(Forecaster):
    """Return a constant point forecast, plus quantiles when requested."""

    def __init__(self, alias: str = "Constant", value: float = 1.0):
        self.alias = alias
        self.value = value

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        last_times = df.groupby("unique_id")["ds"].max()
        fcst = make_future_dataframe(
            uids=last_times.index.tolist(),
            last_times=last_times,
            h=h,
            freq=freq,
        )
        fcst[self.alias] = self.value
        if qc.quantiles:
            for q in qc.quantiles:
                fcst[f"{self.alias}-q-{int(q * 100)}"] = fcst[self.alias] + q
            fcst = qc.maybe_convert_quantiles_to_level(fcst, models=[self.alias])
        return fcst


class SeasonalNaiveModel(Forecaster):
    """Repeat the last seasonal period, mimicking a real model's output shape."""

    alias = "SeasonalNaive"

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        results = []
        for uid, group in df.groupby("unique_id"):
            y = group["y"].to_numpy()
            effective_length = min(season_length, len(y))
            future = make_future_dataframe(
                uids=[uid],
                last_times=pd.Series([group["ds"].max()], index=[uid]),
                h=h,
                freq=freq,
            )
            seasonal_values = y[-effective_length:]
            future[self.alias] = seasonal_values[np.arange(h) % effective_length]
            results.append(future)
        fcst = pd.concat(results, ignore_index=True)
        if qc.quantiles:
            for q in qc.quantiles:
                fcst[f"{self.alias}-q-{int(q * 100)}"] = fcst[self.alias] * (1 + q)
            fcst = qc.maybe_convert_quantiles_to_level(fcst, models=[self.alias])
        return fcst
