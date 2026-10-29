import numpy as np
import pandas as pd
import pytest
from utilsforecast.processing import make_future_dataframe

import foundationforecast.models as ff_models
from foundationforecast import FoundationForecast
from foundationforecast.core.conformal import ConformalIntervals
from foundationforecast.core.forecaster import ExogCapableForecaster, Forecaster
from foundationforecast.core.utils import process_panel_from_df


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
        if panel is None:
            panel = process_panel_from_df(df)
        futr = make_future_dataframe(
            uids=panel.uids,
            last_times=pd.to_datetime(panel.last_times),
            freq=freq,
            h=h,
            id_col="unique_id",
            time_col="ds",
        )
        futr[self.alias] = np.repeat([arr.mean() for arr in panel.series_arrays], h)
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


def test_conformal_forecast_through_foundation_forecast_matches_direct():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(n_series=3, length=40)
    direct = model.forecast(df=df, h=3, freq="D", level=[80])
    via_ff = FoundationForecast(models=[model]).forecast(
        df=df, h=3, freq="D", level=[80]
    )
    pd.testing.assert_frame_equal(
        direct.reset_index(drop=True),
        via_ff[direct.columns].reset_index(drop=True),
    )


def test_conformal_forecast_with_integer_unique_ids():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(length=40)
    df["unique_id"] = df["unique_id"].astype(int)
    fcst = model.forecast(df=df, h=3, freq="D", level=[80])
    assert fcst["Mean-lo-80"].notna().all()
    assert fcst["Mean-hi-80"].notna().all()


@pytest.mark.parametrize(
    "model_cls",
    [
        getattr(ff_models, name)
        for name in ff_models.__all__
        if isinstance(getattr(ff_models, name), type)
        and issubclass(getattr(ff_models, name), Forecaster)
    ],
)
def test_model_constructors_accept_prediction_intervals(model_cls):
    intervals = ConformalIntervals(n_windows=3)
    model = model_cls(prediction_intervals=intervals)
    assert model.prediction_intervals is intervals


def test_native_level_without_prediction_intervals():
    model = _MeanForecaster()
    df = _panel(length=10)
    fcst = model.forecast(df=df, h=2, freq="D", level=[80])
    assert "Mean-lo-80" not in fcst.columns
