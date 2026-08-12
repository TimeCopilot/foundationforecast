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


def test_prepare_level_and_quantiles_with_levels():
    qc = QuantileConverter(level=[80, 95])
    assert qc.level == [80, 95]
    assert qc.level_was_provided


@pytest.mark.parametrize(
    "quantiles,expected_level",
    [
        ([0.1, 0.5, 0.9], [0, 80]),
        ([0.1, 0.5, 0.2, 0.9], [0, 60, 80]),
        ([0.5], [0]),
    ],
)
def test_prepare_level_and_quantiles_with_quantiles(quantiles, expected_level):
    qc = QuantileConverter(level=None, quantiles=quantiles)
    assert qc.quantiles == quantiles
    assert qc.level == expected_level
    assert not qc.level_was_provided


def test_prepare_level_and_quantiles_error_both():
    with pytest.raises(ValueError):
        QuantileConverter(level=[90], quantiles=[0.9])


@pytest.mark.parametrize(
    "n_models,quantiles",
    [
        (1, [0.1]),
        (2, [0.1, 0.5, 0.9]),
        (2, [0.1, 0.5, 0.2, 0.9]),
    ],
)
def test_maybe_convert_level_to_quantiles(n_models, quantiles):
    models = [f"model{i}" for i in range(n_models)]
    qc = QuantileConverter(quantiles=quantiles)
    assert not qc.level_was_provided
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=10,
        n_models=n_models,
        level=qc.level,
    )
    result_df = qc.maybe_convert_level_to_quantiles(
        df,
        models=models,
    )
    exp_n_cols = 3 + (1 + len(quantiles)) * n_models
    assert result_df.shape[1] == exp_n_cols
    for model in models:
        assert qc.quantiles is not None
        for q in qc.quantiles:
            assert f"{model}-q-{int(q * 100)}" in result_df.columns
        if 0.5 in qc.quantiles:
            pd.testing.assert_series_equal(
                result_df[f"{model}-q-50"],
                result_df[f"{model}"],
                check_names=False,
            )
    pd.testing.assert_frame_equal(
        df,
        qc.maybe_convert_quantiles_to_level(df, models=models),
    )


@pytest.mark.parametrize(
    "n_models,level",
    [
        (1, [80]),
        (2, [0, 80]),
        (2, [60, 80]),
    ],
)
def test_maybe_convert_quantiles_to_level(n_models, level):
    models = [f"model{i}" for i in range(n_models)]
    qc = QuantileConverter(level=level)
    assert qc.level_was_provided
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=10,
        n_models=n_models,
    )
    for model in models:
        for q in qc.quantiles:  # type: ignore
            df[f"{model}-q-{int(q * 100)}"] = q
    result_df = qc.maybe_convert_quantiles_to_level(
        df,
        models=models,
    )
    exp_n_cols = 3 + (1 + len(level) * 2) * n_models
    assert result_df.shape[1] == exp_n_cols
    for model in models:
        for lv in level:
            if lv == 0:
                pd.testing.assert_series_equal(
                    result_df[model],
                    df[f"{model}-q-50"],
                    check_names=False,
                )
            else:
                alpha = round(1 - lv / 100, 2)
                q_lo = int((alpha / 2) * 100)
                q_hi = int((1 - alpha / 2) * 100)
                pd.testing.assert_series_equal(
                    result_df[f"{model}-lo-{lv}"],
                    df[f"{model}-q-{q_lo}"],
                    check_names=False,
                )
                pd.testing.assert_series_equal(
                    result_df[f"{model}-hi-{lv}"],
                    df[f"{model}-q-{q_hi}"],
                    check_names=False,
                )
    pd.testing.assert_frame_equal(
        df,
        qc.maybe_convert_level_to_quantiles(df, models=models),
    )


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
