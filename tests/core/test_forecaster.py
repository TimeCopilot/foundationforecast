import pandas as pd
import pytest

from tests.helpers import (
    DummyModel,
    SeasonalNaiveModel,
    generate_series,
    generate_series_with_anomalies,
)
from foundationforecast.core.forecaster import (
    QuantileConverter,
    get_seasonality,
    maybe_infer_freq,
)


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
    # Controlled y pattern required for exact seasonal assertion
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
    # Controlled y pattern required for exact seasonal assertion
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
        ([0.1, 0.5, 0.9], [80]),
        ([0.1, 0.5, 0.2, 0.9], [60, 80]),
        ([0.5], None),
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
    result = model.detect_anomalies(df, h=h, freq="D")
    assert len(result) > 0
    if h is not None:
        cutoffs = result["cutoff"].unique()
        assert len(cutoffs) >= 1


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("n_windows", [None, 1, 3])
def test_detect_anomalies_n_windows_parameter(model, n_windows):
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
    result = model.detect_anomalies(df, n_windows=n_windows, freq="D")
    assert len(result) > 0
    actual_windows = len(result["cutoff"].unique())
    if n_windows is not None:
        assert actual_windows <= n_windows


@pytest.mark.parametrize("model", [SeasonalNaiveModel()])
@pytest.mark.parametrize("level", [80, 95, 99])
def test_detect_anomalies_confidence_level(model, level):
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
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
    # Exactly 2 rows required to trigger "series too short" error
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
