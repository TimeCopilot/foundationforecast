import pandas as pd
import pytest
from utilsforecast.processing import make_future_dataframe

from tests.helpers import DummyModel, generate_panel_with_futr_exog, generate_series
from foundationforecast import FoundationForecast
from foundationforecast.core.forecaster import ExogCapableForecaster, Forecaster
from foundationforecast.core.quantiles import quantile_column_name
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
            assert quantile_column_name(model.alias, q) in fcst_df.columns


def test_foundation_forecast_fallback_model():
    class FailingModel(ExogCapableForecaster):
        alias = "FailingModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = df, h, freq, level, quantiles, panel
            raise RuntimeError("Intentional failure")

    class FallbackModel(ExogCapableForecaster):
        alias = "FallbackModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = panel
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


def test_foundation_forecast_omits_exog_kwargs_for_legacy_signature():
    """Forecaster without X_df kwargs works through FoundationForecast."""

    class LegacySignatureModel(Forecaster):
        alias = "LegacySignatureModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None, panel=None):
            _ = level, quantiles, panel
            freq = self._maybe_infer_freq(df, freq)
            last_times = df.groupby("unique_id")["ds"].max()
            fcst = make_future_dataframe(
                uids=last_times.index.tolist(),
                last_times=last_times,
                h=h,
                freq=freq,
            )
            fcst[self.alias] = 0.0
            return fcst

    df = generate_series(n_series=1, freq="D", min_length=10)
    fcst_df = FoundationForecast(models=[LegacySignatureModel()]).forecast(
        df=df, h=2, freq="D"
    )
    assert "LegacySignatureModel" in fcst_df.columns
    assert len(fcst_df) == 2


def test_foundation_forecast_fallback_with_exog_capable_forecaster():
    class LegacyFailingModel(ExogCapableForecaster):
        alias = "LegacyFailingModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = df, h, freq, level, quantiles, panel
            raise RuntimeError("Intentional failure")

    class LegacyFallbackModel(ExogCapableForecaster):
        alias = "LegacyFallbackModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = panel
            n = len(df["unique_id"].unique()) * h
            return pd.DataFrame(
                {
                    "unique_id": ["A"] * n,
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "LegacyFallbackModel": range(n),
                }
            )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(
        models=[LegacyFailingModel()],
        fallback_model=LegacyFallbackModel(),
    )
    fcst_df = forecaster.forecast(df=df, h=2, freq="D")
    assert "LegacyFailingModel" in fcst_df.columns
    assert len(fcst_df) == 2


def test_foundation_forecast_no_fallback_raises():
    class FailingModel(ExogCapableForecaster):
        alias = "FailingModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = df, h, freq, level, quantiles, panel
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


def test_foundation_forecast_validates_exog_before_fallback():
    class NonNativeExogModel(ExogCapableForecaster):
        alias = "NonNativeExogModel"

        def supports_native_futr_exog(self) -> bool:
            return False

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = df, h, freq, level, quantiles, panel
            raise RuntimeError("should not run")

    class FallbackModel(ExogCapableForecaster):
        alias = "FallbackModel"

        def _forecast_univariate(
            self, df, h, freq=None, level=None, quantiles=None, panel=None
        ):
            _ = panel
            n = len(df["unique_id"].unique()) * h
            return pd.DataFrame(
                {
                    "unique_id": ["A"] * n,
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "FallbackModel": range(n),
                }
            )

    df, X_df, _ = generate_panel_with_futr_exog(n_series=1, freq="D", h=3)
    forecaster = FoundationForecast(
        models=[NonNativeExogModel()],
        fallback_model=FallbackModel(),
    )
    with pytest.raises(ValueError, match="does not support known-future"):
        forecaster.forecast(df=df, h=3, freq="D", X_df=X_df)
