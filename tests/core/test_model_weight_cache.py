import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.model_weight_cache import (
    ModelWeightCache,
    get_model_weight_cache,
    reset_model_weight_cache,
    set_max_cached_models,
)
from foundationforecast.models.chronos import Chronos


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_model_weight_cache()
    yield
    reset_model_weight_cache()


def test_model_weight_cache_lru_eviction():
    cache = ModelWeightCache(max_cached_models=2)
    loads = []

    def loader(name: str):
        def _load():
            loads.append(name)
            return f"model-{name}"

        return _load

    assert cache.get_or_load("a", loader("a")) == "model-a"
    assert cache.get_or_load("b", loader("b")) == "model-b"
    assert cache.get_or_load("a", loader("a")) == "model-a"
    assert loads == ["a", "b"]

    cache.get_or_load("c", loader("c"))
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache
    assert loads == ["a", "b", "c"]


def test_clear_prefix_does_not_match_sibling_repo_ids():
    cache = ModelWeightCache(max_cached_models=3)
    cache.get_or_load("T0:repo-a:float32", lambda: "model-a")
    cache.get_or_load("T0:repo-a2:float32", lambda: "model-a2")
    cache.clear_prefix("T0:repo-a")
    assert "T0:repo-a:float32" not in cache
    assert "T0:repo-a2:float32" in cache


def test_zero_capacity_reloads_each_call(mocker):
    load_count = 0
    fake_model = mocker.Mock()

    def fake_load():
        nonlocal load_count
        load_count += 1
        return fake_model

    df = _patch_chronos_forecast_deps(mocker, fake_load)
    set_max_cached_models(0)
    model = Chronos(repo_id="amazon/chronos-t5-tiny", reuse_loaded_model=True)

    model.forecast(df=df, h=2, freq="D")
    model.forecast(df=df, h=2, freq="D")
    assert load_count == 2


def test_set_max_cached_models_evicts_when_shrinking():
    set_max_cached_models(2)
    cache = get_model_weight_cache()
    cache.get_or_load("a", lambda: "model-a")
    cache.get_or_load("b", lambda: "model-b")
    set_max_cached_models(1)
    cache = get_model_weight_cache()
    assert "a" not in cache
    assert "b" in cache


def _patch_chronos_forecast_deps(mocker, load_side_effect):
    mocker.patch.object(Chronos, "_load_model", side_effect=load_side_effect)
    mocker.patch.object(
        Chronos,
        "_predict",
        return_value=(np.array([1.0, 2.0]), None),
    )
    dataset = mocker.Mock()
    dataset.make_future_dataframe.return_value = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2020-01-01", periods=2, freq="D"),
        }
    )
    mocker.patch(
        "foundationforecast.models.chronos.TimeSeriesDataset.from_df",
        return_value=dataset,
    )
    return pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10, freq="D"),
            "y": np.arange(10, dtype=float),
        }
    )


def test_chronos_reuses_cached_model(mocker):
    load_count = 0
    fake_model = mocker.Mock()

    def fake_load():
        nonlocal load_count
        load_count += 1
        return fake_model

    df = _patch_chronos_forecast_deps(mocker, fake_load)
    model = Chronos(repo_id="amazon/chronos-t5-tiny", reuse_loaded_model=True)

    model.forecast(df=df, h=2, freq="D")
    model.forecast(df=df, h=2, freq="D")
    assert load_count == 1


def test_chronos_bypasses_cache_when_disabled(mocker):
    load_count = 0
    fake_model = mocker.Mock()

    def fake_load():
        nonlocal load_count
        load_count += 1
        return fake_model

    df = _patch_chronos_forecast_deps(mocker, fake_load)
    model = Chronos(repo_id="amazon/chronos-t5-tiny", reuse_loaded_model=False)

    model.forecast(df=df, h=2, freq="D")
    model.forecast(df=df, h=2, freq="D")
    assert load_count == 2


def test_chronos_clear_model_cache(mocker):
    load_count = 0
    fake_model = mocker.Mock()

    def fake_load():
        nonlocal load_count
        load_count += 1
        return fake_model

    df = _patch_chronos_forecast_deps(mocker, fake_load)
    model = Chronos(repo_id="amazon/chronos-t5-tiny")

    model.forecast(df=df, h=2, freq="D")
    model.clear_model_cache()
    model.forecast(df=df, h=2, freq="D")
    assert load_count == 2


def test_foundation_forecast_clean_cache_calls_clear_model_cache(mocker):
    from tests.helpers import generate_series
    from foundationforecast import FoundationForecast

    model_a = Chronos(repo_id="amazon/chronos-t5-tiny", alias="ChronosA")
    model_b = Chronos(repo_id="amazon/chronos-bolt-tiny", alias="ChronosB")
    clear_calls = {"a": 0, "b": 0}

    def clear_a():
        clear_calls["a"] += 1

    def clear_b():
        clear_calls["b"] += 1

    mocker.patch.object(model_a, "clear_model_cache", side_effect=clear_a)
    mocker.patch.object(model_b, "clear_model_cache", side_effect=clear_b)
    fcst = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.date_range("2020-01-01", periods=2, freq="D"),
            "ChronosA": [1.0, 2.0],
        }
    )
    fcst_b = fcst.rename(columns={"ChronosA": "ChronosB"})
    mocker.patch.object(model_a, "forecast", return_value=fcst)
    mocker.patch.object(model_b, "forecast", return_value=fcst_b)

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(
        models=[model_a, model_b],
        clean_cache=True,
    )
    forecaster.forecast(df=df, h=2, freq="D")
    assert clear_calls == {"a": 2, "b": 2}


def test_foundation_forecast_clean_cache_includes_fallback(mocker):
    from tests.helpers import generate_series
    from foundationforecast import FoundationForecast
    from foundationforecast.core.forecaster import Forecaster

    class FailingModel(Forecaster):
        alias = "FailingModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            raise RuntimeError("Intentional failure")

    fallback = Chronos(repo_id="amazon/chronos-t5-tiny", alias="Fallback")
    clear_calls = {"fallback": 0}

    def clear_fallback():
        clear_calls["fallback"] += 1

    mocker.patch.object(fallback, "clear_model_cache", side_effect=clear_fallback)
    mocker.patch.object(
        fallback,
        "forecast",
        return_value=pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": pd.date_range("2020-01-01", periods=2, freq="D"),
                "Fallback": [1.0, 2.0],
            }
        ),
    )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = FoundationForecast(
        models=[FailingModel()],
        fallback_model=fallback,
        clean_cache=True,
    )
    forecaster.forecast(df=df, h=2, freq="D")
    assert clear_calls["fallback"] == 1
