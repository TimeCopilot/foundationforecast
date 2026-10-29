import pandas as pd
import pytest
from utilsforecast.processing import make_future_dataframe

from foundationforecast.core.conformal import ConformalIntervals
from foundationforecast.core.forecaster import ExogCapableForecaster


class _MeanForecaster(ExogCapableForecaster):
    """Deterministic forecaster: point forecast = training mean per series."""

    def __init__(
        self,
        *,
        alias: str = "Mean",
        prediction_intervals: ConformalIntervals | None = None,
    ) -> None:
        super().__init__(prediction_intervals=prediction_intervals)
        self.alias = alias

    def _forecast_univariate(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        level,
        quantiles,
        panel,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        futr = make_future_dataframe(
            uids=df["unique_id"].unique(),
            last_times=df.groupby("unique_id")["ds"].max(),
            freq=freq,
            h=h,
            id_col="unique_id",
            time_col="ds",
        )
        means = df.groupby("unique_id")["y"].mean()
        futr[self.alias] = futr["unique_id"].map(means)
        return futr


def _panel(n_series: int = 2, length: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        for t in range(length):
            rows.append(
                {
                    "unique_id": str(i),
                    "ds": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                    "y": float(i + 1 + (t % 5)),
                }
            )
    return pd.DataFrame(rows)


def test_conformal_forecast_requires_level():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    with pytest.raises(ValueError, match="level` or `quantiles`"):
        model.forecast(_panel(), h=3, freq="D")


def test_conformal_forecast_adds_interval_columns():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(length=40)
    fcst = model.forecast(df=df, h=3, freq="D", level=[80, 90])
    assert "Mean-lo-80" in fcst.columns
    assert "Mean-hi-90" in fcst.columns
    assert not any("-q-" in c for c in fcst.columns)


def test_native_level_without_prediction_intervals():
    model = _MeanForecaster()
    df = _panel(length=10)
    fcst = model.forecast(df=df, h=2, freq="D", level=[80])
    assert "Mean-lo-80" not in fcst.columns
