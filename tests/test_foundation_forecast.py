import pandas as pd
import pytest

from tests.helpers import DummyModel, generate_series
from foundationforecast import FoundationForecast
from foundationforecast.core.forecaster import Forecaster
from foundationforecast.models.moirai import Moirai


@pytest.fixture
def models():
    return [DummyModel(alias="DummyA"), DummyModel(alias="DummyB")]


def test_foundation_forecast_requires_unique_aliases():
    with pytest.raises(ValueError, match="Duplicate model aliases"):
        FoundationForecast(models=[DummyModel(), DummyModel()])


def test_foundation_forecast_has_forecast_method():
    ff = FoundationForecast(models=[DummyModel()])
    assert hasattr(ff, "forecast")


@pytest.mark.parametrize(
    "freq,h",
    [
        ("D", 2),
        ("W-MON", 3),
    ],
)
def test_foundation_forecast_forecast(models, freq, h):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = FoundationForecast(models=models)
    fcst_df = forecaster.forecast(df=df, h=h, freq=freq)
    assert len(fcst_df.columns) == 2 + len(models)
    assert len(fcst_df) == h * n_uids
    for model in models:
        assert model.alias in fcst_df.columns


@pytest.mark.parametrize(
    "freq,h,n_windows,step_size",
    [
        ("D", 2, 2, 1),
        ("W-MON", 3, 2, 2),
    ],
)
def test_foundation_forecast_cross_validation(models, freq, h, n_windows, step_size):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = FoundationForecast(models=models)
    fcst_df = forecaster.cross_validation(
        df=df,
        h=h,
        freq=freq,
        n_windows=n_windows,
        step_size=step_size,
    )
    assert len(fcst_df.columns) == 4 + len(models)
    uids = df["unique_id"].unique()
    for uid in uids:  # noqa: B007
        fcst_df_uid = fcst_df.query("unique_id == @uid")
        assert fcst_df_uid["cutoff"].nunique() == n_windows
        assert len(fcst_df_uid) == n_windows * h
    for model in models:
        assert model.alias in fcst_df.columns


def test_foundation_forecast_forecast_with_level(models):
    n_uids = 3
    level = [80, 90]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = FoundationForecast(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", level=level)  # type: ignore
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + 2 * len(level))
    for model in models:
        assert model.alias in fcst_df.columns
        for lv in level:
            assert f"{model.alias}-lo-{lv}" in fcst_df.columns
            assert f"{model.alias}-hi-{lv}" in fcst_df.columns


def test_foundation_forecast_forecast_with_quantiles(models):
    n_uids = 3
    quantiles = [0.1, 0.9]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = FoundationForecast(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", quantiles=quantiles)
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + len(quantiles))
    for model in models:
        assert model.alias in fcst_df.columns
        for q in quantiles:
            assert f"{model.alias}-q-{int(100 * q)}" in fcst_df.columns


def test_foundation_forecast_fallback_model():
    class FailingModel(Forecaster):
        alias = "FailingModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            raise RuntimeError("Intentional failure")

    class FallbackModel(Forecaster):
        alias = "FallbackModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            n = len(df["unique_id"].unique()) * h
            return pd.DataFrame(
                {
                    "unique_id": ["A"] * n,
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "FallbackModel": range(n),
                }
            )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(
        models=[FailingModel()],
        fallback_model=FallbackModel(),
    )
    fcst_df = forecaster.forecast(df=df, h=2, freq="D")
    assert "FailingModel" in fcst_df.columns
    assert "FallbackModel" not in fcst_df.columns
    assert len(fcst_df) == 2


def test_foundation_forecast_no_fallback_raises():
    class FailingModel(Forecaster):
        alias = "FailingModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            raise RuntimeError("Intentional failure")

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(models=[FailingModel()])
    with pytest.raises(RuntimeError, match="Intentional failure"):
        forecaster.forecast(df=df, h=2, freq="D")


def test_foundation_forecast_unique_aliases_works():
    forecaster = FoundationForecast(
        models=[DummyModel(alias="DummyA"), DummyModel(alias="DummyB")]
    )
    assert len(forecaster.models) == 2
    assert forecaster.models[0].alias == "DummyA"
    assert forecaster.models[1].alias == "DummyB"


def test_foundation_forecast_mixed_models_unique_aliases():
    forecaster = FoundationForecast(
        models=[
            DummyModel(alias="DummyA"),
            DummyModel(alias="DummyB"),
            DummyModel(alias="DummyC"),
        ]
    )
    assert len(forecaster.models) == 3


def test_foundation_forecast_clean_cache_runs_after_each_model(monkeypatch, models):
    calls = []

    monkeypatch.setattr(
        FoundationForecast,
        "_clean_model_cache",
        staticmethod(lambda: calls.append("cleaned")),
    )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(models=models, clean_cache=True)
    forecaster.forecast(df=df, h=2, freq="D")
    assert calls == ["cleaned"] * len(models)


def test_foundation_forecast_duplicate_aliases_with_moirai():
    model1 = Moirai(repo_id="Salesforce/moirai-1.0-R-small", alias="Moirai")
    model2 = Moirai(repo_id="Salesforce/moirai-1.0-R-large", alias="Moirai")

    with pytest.raises(
        ValueError, match="Duplicate model aliases found: \\['Moirai'\\]"
    ):
        FoundationForecast(models=[model1, model2])


def test_foundation_forecast_rejects_empty_models():
    with pytest.raises(ValueError, match="At least one model is required"):
        FoundationForecast(models=[])


def test_foundation_forecast_validates_input():
    df = generate_series(n_series=1, freq="D", min_length=5, max_length=5)
    forecaster = FoundationForecast(models=[DummyModel()])
    with pytest.raises(ValueError, match="h must be a positive integer"):
        forecaster.forecast(df=df, h=0, freq="D")
