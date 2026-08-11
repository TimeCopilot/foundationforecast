import pandas as pd
import pytest
from utilsforecast.data import generate_series
from utilsforecast.processing import make_future_dataframe

from foundationforecast.core.forecaster import (
    Forecaster,
    QuantileConverter,
    get_seasonality,
    maybe_infer_freq,
)


class DummyModel(Forecaster):
    alias = "dummy"

    def forecast(self, df, h, freq=None, level=None, quantiles=None):
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
        return fcst


def test_quantile_converter_rejects_level_and_quantiles_together():
    with pytest.raises(ValueError, match="must not provide both"):
        QuantileConverter(level=[80, 95], quantiles=[0.1, 0.9])


def test_quantile_converter_level_to_quantiles():
    qc = QuantileConverter(level=[80])
    assert qc.quantiles == [0.1, 0.9]
    assert qc.level_was_provided is True


def test_quantile_converter_quantiles_to_level():
    qc = QuantileConverter(quantiles=[0.1, 0.9])
    assert qc.level == [80]
    assert qc.level_was_provided is False


def test_maybe_infer_freq_daily():
    df = generate_series(n_series=2, freq="D", min_length=20, max_length=20)
    assert maybe_infer_freq(df, None) == "D"


def test_get_seasonality_daily_default():
    assert get_seasonality("D") == 1


def test_get_seasonality_custom():
    assert get_seasonality("D", custom_seasonalities={"D": 7}) == 7


def test_forecaster_cross_validation():
    df = generate_series(n_series=1, freq="D", min_length=30, max_length=30)
    model = DummyModel()
    cv = model.cross_validation(df=df, h=7, n_windows=2)
    assert {"unique_id", "ds", "cutoff", "y", "dummy"}.issubset(cv.columns)
    assert cv["cutoff"].nunique() == 2


def test_detect_anomalies_short_series_error():
    model = DummyModel()
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2023-01-01", periods=2, freq="D"),
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Cannot perform anomaly detection"):
        model.detect_anomalies(df, h=5, freq="D")


def test_detect_anomalies_basic():
    df = generate_series(n_series=1, freq="D", min_length=30, max_length=30)
    model = DummyModel()
    result = model.detect_anomalies(df, h=7, freq="D")
    assert f"{model.alias}-anomaly" in result.columns
    assert len(result) > 0
