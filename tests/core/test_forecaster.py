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


class SeasonalNaiveModel(Forecaster):
    alias = "SeasonalNaive"

    def __init__(self, season_length: int | None = None):
        self.season_length = season_length

    def forecast(self, df, h, freq=None, level=None, quantiles=None):
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


def test_get_seasonality_custom_seasonalities():
    assert get_seasonality("D", custom_seasonalities={"D": 7}) == 7
    assert get_seasonality("D", custom_seasonalities={"D": 7}) == 7
    assert get_seasonality("D") == 1


@pytest.mark.parametrize("freq", ["MS", "W-MON", "D"])
def test_maybe_infer_freq(freq):
    df = generate_series(
        n_series=2,
        freq=freq,
    )
    assert maybe_infer_freq(df, None) == freq
    assert maybe_infer_freq(df, "H") == "H"


def test_maybe_get_seasonality_explicit():
    model = SeasonalNaiveModel(season_length=4)
    assert model._maybe_get_seasonality("D") == 4


@pytest.mark.parametrize("freq", ["M", "MS", "W-MON", "D"])
def test_maybe_get_seasonality_infer(freq):
    model = SeasonalNaiveModel(season_length=None)
    assert model._maybe_get_seasonality(freq) == get_seasonality(freq)


@pytest.mark.parametrize("freq", ["M", "MS", "W-MON", "D"])
def test_get_seasonality_inferred_correctly(freq):
    season_length = get_seasonality(freq)
    y = 2 * list(range(1, season_length + 1))
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * len(y),
            "ds": pd.date_range("2023-01-01", periods=len(y), freq=freq),
            "y": y,
        }
    )
    model = SeasonalNaiveModel()
    fcst = model.forecast(df, h=season_length, freq=freq)
    assert (fcst["SeasonalNaive"].values == y[-season_length:]).all()


@pytest.mark.parametrize("season_length,freq", [(4, "D"), (7, "W-MON")])
def test_seasonality_used_correctly(season_length, freq):
    y = 2 * list(range(1, season_length + 1))
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * len(y),
            "ds": pd.date_range("2023-01-01", periods=len(y), freq=freq),
            "y": y,
        }
    )
    model = SeasonalNaiveModel(season_length=season_length)
    fcst = model.forecast(df, h=season_length, freq=freq)
    assert (fcst["SeasonalNaive"].values == y[-season_length:]).all()


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
    df["unique_id"] = df["unique_id"].astype(str)

    if anomaly_positions is not None:
        for series_id in df["unique_id"].unique():
            series_data = df[df["unique_id"] == series_id].copy()
            for pos in anomaly_positions:
                if pos < len(series_data):
                    anomaly_idx = series_data.index[pos]
                    df.loc[anomaly_idx, "y"] += anomaly_magnitude

    return df


def test_forecaster_cross_validation():
    df = generate_series(n_series=1, freq="D", min_length=30, max_length=30)
    model = DummyModel()
    cv = model.cross_validation(df=df, h=7, n_windows=2)
    assert {"unique_id", "ds", "cutoff", "y", "dummy"}.issubset(cv.columns)
    assert cv["cutoff"].nunique() == 2


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("freq", ["D", "H", "W-MON"])
def test_detect_anomalies_basic_functionality(model, freq):
    df = generate_series(n_series=2, freq=freq, min_length=30, max_length=30)
    df["unique_id"] = df["unique_id"].astype(str)
    result = model.detect_anomalies(df)
    assert len(result) > 0
    expected_cols = [
        "unique_id",
        "ds",
        "cutoff",
        "y",
        model.alias,
        f"{model.alias}-lo-99",
        f"{model.alias}-hi-99",
        f"{model.alias}-anomaly",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
    anomaly_col_name = (
        "anomaly" if "anomaly" in result.columns else f"{model.alias}-anomaly"
    )
    assert result[anomaly_col_name].dtype == bool
    assert pd.api.types.is_numeric_dtype(result[f"{model.alias}-lo-99"])
    assert pd.api.types.is_numeric_dtype(result[f"{model.alias}-hi-99"])


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
def test_detect_anomalies_with_artificial_anomalies(model):
    df = generate_series_with_anomalies(
        n_series=2,
        freq="D",
        min_length=50,
        max_length=50,
        anomaly_positions=[45, 47],
        anomaly_magnitude=10.0,
    )
    result = model.detect_anomalies(df, freq="D", level=95)
    anomaly_col = f"{model.alias}-anomaly"
    assert anomaly_col in result.columns
    detected_anomalies = result[anomaly_col].sum()
    assert detected_anomalies >= 0
    anomalies = result[result[anomaly_col]]
    lo_col = f"{model.alias}-lo-95"
    hi_col = f"{model.alias}-hi-95"
    for _, row in anomalies.iterrows():
        assert row["y"] < row[lo_col] or row["y"] > row[hi_col], (
            f"Anomaly at index {row.name} is not outside the interval: "
            f"{row[lo_col]} <= {row['y']} <= {row[hi_col]}"
        )


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("h", [None, 3, 7])
def test_detect_anomalies_horizon_parameter(model, h):
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
    df["unique_id"] = df["unique_id"].astype(str)
    result = model.detect_anomalies(df, h=h, freq="D")
    assert len(result) > 0
    if h is not None:
        cutoffs = result["cutoff"].unique()
        assert len(cutoffs) >= 1


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("n_windows", [None, 1, 3])
def test_detect_anomalies_n_windows_parameter(model, n_windows):
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
    df["unique_id"] = df["unique_id"].astype(str)
    result = model.detect_anomalies(df, n_windows=n_windows, freq="D")
    assert len(result) > 0
    actual_windows = len(result["cutoff"].unique())
    if n_windows is not None:
        assert actual_windows <= n_windows


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("level", [80, 95, 99])
def test_detect_anomalies_confidence_level(model, level):
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
    df["unique_id"] = df["unique_id"].astype(str)
    result = model.detect_anomalies(df, level=level, freq="D")
    lo_col = f"{model.alias}-lo-{level}"
    hi_col = f"{model.alias}-hi-{level}"

    assert lo_col in result.columns
    assert hi_col in result.columns
    assert (result[lo_col] <= result[hi_col]).all()
    anomalies = result[result[f"{model.alias}-anomaly"]]
    for _, row in anomalies.iterrows():
        assert row["y"] < row[lo_col] or row["y"] > row[hi_col], (
            f"Anomaly at index {row.name} is not outside the interval: "
            f"{row[lo_col]} <= {row['y']} <= {row[hi_col]}"
        )


def test_detect_anomalies_short_series_error():
    model = SeasonalNaiveModel()
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2023-01-01", periods=2, freq="D"),
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Cannot perform anomaly detection"):
        model.detect_anomalies(df, h=5, freq="D")
