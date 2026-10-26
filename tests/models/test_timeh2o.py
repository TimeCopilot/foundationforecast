import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

if sys.version_info < (3, 12):
    pytest.skip("TimeH2O requires Python >= 3.12", allow_module_level=True)

import pandas as pd  # noqa: E402

from tests.helpers import generate_series  # noqa: E402
from foundationforecast.models.timeh2o import TimeH2O  # noqa: E402


def _fake_forecast_many(contexts, horizon, quantile_levels, **kwargs):
    """Each quantile column holds its own level, so mappings are checkable."""
    levels = np.asarray(quantile_levels, dtype=np.float32)
    return [SimpleNamespace(quantiles=np.tile(levels, (horizon, 1))) for _ in contexts]


def _mock_model(mocker, model: TimeH2O) -> MagicMock:
    backend = MagicMock()
    backend.forecast_many.side_effect = _fake_forecast_many
    cm = MagicMock()
    cm.__enter__.return_value = backend
    cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=cm)
    return backend


def test_timeh2o_passes_requested_quantiles_with_median(mocker):
    model = TimeH2O(batch_size=2)
    backend = _mock_model(mocker, model)
    df = generate_series(3, freq="D", min_length=32, max_length=32)

    quantiles = [0.05, 0.25, 0.95]
    fcst = model.forecast(df, h=4, freq="D", quantiles=quantiles)

    for call in backend.forecast_many.call_args_list:
        assert call.kwargs["quantile_levels"] == [0.05, 0.25, 0.5, 0.95]
    assert len(fcst) == 3 * 4
    np.testing.assert_allclose(fcst["TimeH2O"], 0.5)
    for q in quantiles:
        np.testing.assert_allclose(fcst[f"TimeH2O-q-{int(q * 100)}"], q)


def test_timeh2o_level_maps_to_interval_columns(mocker):
    model = TimeH2O(batch_size=4)
    _mock_model(mocker, model)
    df = generate_series(2, freq="D", min_length=32, max_length=32)

    fcst = model.forecast(df, h=3, freq="D", level=[80])

    np.testing.assert_allclose(fcst["TimeH2O-lo-80"], 0.1)
    np.testing.assert_allclose(fcst["TimeH2O-hi-80"], 0.9)


def test_timeh2o_truncates_context(mocker):
    model = TimeH2O(context_length=16, batch_size=4)
    backend = _mock_model(mocker, model)
    df = generate_series(2, freq="D", min_length=10, max_length=40)

    model.forecast(df, h=2, freq="D")

    contexts = backend.forecast_many.call_args.args[0]
    lengths = df.groupby("unique_id").size().to_numpy()
    assert [len(c) for c in contexts] == list(np.minimum(lengths, 16))
    assert backend.forecast_many.call_args.kwargs["context_length"] == 16


def test_timeh2o_native_futr_exog_aligns_covariates(mocker):
    model = TimeH2O(context_length=20, batch_size=2)
    backend = _mock_model(mocker, model)
    n, h = 32, 3
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": ds,
            "y": np.arange(n, dtype=float),
            "x1": np.arange(n, dtype=float),
            "x2": -np.arange(n, dtype=float),
        }
    )
    X_df = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=h, freq="D"),
            "x1": np.arange(n, n + h, dtype=float),
            "x2": -np.arange(n, n + h, dtype=float),
        }
    )

    fcst = model.forecast(df, h=h, freq="D", X_df=X_df)

    assert len(fcst) == h
    ((x1, x2),) = backend.forecast_many.call_args.kwargs["future_covariates"]
    assert x1.dtype == np.float32
    np.testing.assert_array_equal(x1, np.arange(n - 20, n + h))
    np.testing.assert_array_equal(x2, -np.arange(n - 20, n + h))


def test_timeh2o_native_futr_exog_keeps_categorical_rows(mocker):
    model = TimeH2O(batch_size=2)
    backend = _mock_model(mocker, model)
    n, h = 10, 2
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": ds,
            "y": np.arange(n, dtype=float),
            "holiday": ["none"] * (n - 1) + ["xmas"],
            "is_weekend": ds.dayofweek >= 5,
        }
    )
    futr_ds = pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    X_df = pd.DataFrame(
        {
            "unique_id": "A",
            "ds": futr_ds,
            "holiday": ["none", "new_year"],
            "is_weekend": futr_ds.dayofweek >= 5,
        }
    )

    model.forecast(df, h=h, freq="D", X_df=X_df)

    ((holiday, is_weekend),) = backend.forecast_many.call_args.kwargs[
        "future_covariates"
    ]
    assert holiday.dtype == object
    assert list(holiday[-3:]) == ["xmas", "none", "new_year"]
    assert is_weekend.dtype == bool
    assert len(is_weekend) == n + h


def test_timeh2o_passes_postprocessing_options(mocker):
    model = TimeH2O(batch_size=2, clip_nonnegative=True, snap_integer=True)
    backend = _mock_model(mocker, model)
    df = generate_series(1, freq="D", min_length=20, max_length=20)

    model.forecast(df, h=2, freq="D")

    kwargs = backend.forecast_many.call_args.kwargs
    assert kwargs["clip_nonnegative"] is True
    assert kwargs["snap_integer"] is True


def test_timeh2o_tiny_has_no_native_futr_exog():
    assert TimeH2O().supports_native_futr_exog()
    assert not TimeH2O(repo_id="h2oai/timeh2o-v1-tiny").supports_native_futr_exog()


@pytest.mark.models
def test_timeh2o_tiny_live_forecast():
    model = TimeH2O(repo_id="h2oai/timeh2o-v1-tiny", batch_size=4)
    df = generate_series(3, freq="D", min_length=48, max_length=96)

    fcst = model.forecast(df, h=5, freq="D", quantiles=[0.1, 0.5, 0.9])

    assert len(fcst) == 3 * 5
    assert fcst["TimeH2O"].notna().all()
    assert (fcst["TimeH2O-q-10"] <= fcst["TimeH2O-q-90"]).all()
