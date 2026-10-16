import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    pytest.skip(
        "T0 requires Python >= 3.11 and < 3.14",
        allow_module_level=True,
    )

import pandas as pd  # noqa: E402

from tests.helpers import generate_series  # noqa: E402
from foundationforecast.models.t0 import T0  # noqa: E402

pytestmark = pytest.mark.models


def test_t0_native_futr_exog_passes_future_covariates(mocker):
    model = T0(context_length=64, batch_size=2, exog_strategy="auto")
    n = 32
    h = 3
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": ds,
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
        }
    )
    futr_ds = pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    X_df = pd.DataFrame(
        {
            "unique_id": ["A"] * h,
            "ds": futr_ds,
            "x1": rng.normal(size=h),
        }
    )

    mock_forecaster = MagicMock()
    mock_out = MagicMock()
    mock_out.quantiles.cpu.return_value.numpy.return_value = np.ones((1, h, 1))
    mock_forecaster.predict.return_value = mock_out

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_forecaster
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)

    fcst = model.forecast(df, h=h, freq="D", X_df=X_df)
    assert fcst.shape[0] == h
    kwargs = mock_forecaster.predict.call_args.kwargs
    assert kwargs["future_covariates"] is not None
    assert kwargs["future_covariates"].shape[0] == 1
    assert kwargs["future_covariates"].shape[1] == 1
    assert kwargs["future_covariates"].shape[2] == min(n, 64) + h


def test_t0_forecast_passes_quantile_levels(mocker):
    model = T0(context_length=64, batch_size=2)
    df = generate_series(2, freq="D", min_length=32, max_length=32)

    mock_forecaster = MagicMock()
    mock_out = MagicMock()
    mock_out.quantiles.cpu.return_value.numpy.return_value = np.ones((2, 3, 3))
    mock_forecaster.predict.return_value = mock_out

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_forecaster
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)

    model.forecast(df, h=3, freq="D", quantiles=[0.1, 0.9])

    kwargs = mock_forecaster.predict.call_args.kwargs
    assert kwargs["quantile_levels"] == [0.1, 0.5, 0.9]
    assert "quantiles" not in kwargs


def test_t0_beta_forecast():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = T0(
        repo_id="theforecastingcompany/t0-beta",
        alias="t0-beta",
        context_length=256,
        batch_size=2,
    )
    fcst = model.forecast(df, h=3, freq="D")
    assert fcst.shape == (6, 3)
    assert "t0-beta" in fcst.columns


def test_t0_beta_quantile_forecast():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = T0(
        repo_id="theforecastingcompany/t0-beta",
        alias="t0-beta",
        context_length=256,
        batch_size=2,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.1, 0.9])
    assert fcst.shape == (6, 5)
    assert "t0-beta" in fcst.columns
    assert "t0-beta-q-10" in fcst.columns
    assert "t0-beta-q-90" in fcst.columns
    assert pd.api.types.is_numeric_dtype(fcst["t0-beta-q-10"])


def _mock_model(mocker, model, quantiles):
    mock_forecaster = MagicMock()
    mock_out = MagicMock()
    mock_out.quantiles.cpu.return_value.numpy.return_value = quantiles
    mock_forecaster.predict.return_value = mock_out
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_forecaster
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)
    return mock_forecaster


def test_t0_forecast_passes_inference_options(mocker):
    model = T0(
        context_length=64,
        batch_size=2,
        trim_leading_missing=True,
        symmetric_averaging=True,
        clip_nonnegative=True,
        snap_integers=True,
    )
    df = generate_series(2, freq="D", min_length=32, max_length=32)
    mock_forecaster = _mock_model(mocker, model, np.ones((2, 3, 1)))

    model.forecast(df, h=3, freq="D")

    kwargs = mock_forecaster.predict.call_args.kwargs
    assert kwargs["trim_leading_missing"] is True
    assert kwargs["symmetric_averaging"] is True
    assert kwargs["clip_nonnegative"] is True
    assert kwargs["snap_integers"] is True
    assert "group_ids" not in kwargs


def test_t0_cross_inference_groups_series(mocker):
    model = T0(context_length=64, batch_size=3, cross_inference=2)
    df = generate_series(3, freq="D", min_length=20, max_length=40)
    mock_forecaster = _mock_model(mocker, model, np.ones((3, 3, 1)))

    fcst = model.forecast(df, h=3, freq="D")

    assert fcst.shape == (9, 3)
    mock_forecaster.predict.assert_called_once()
    kwargs = mock_forecaster.predict.call_args.kwargs
    assert kwargs["group_ids"].tolist() == [0, 0, 1]
    assert kwargs["mask"].shape[0] == 3


def test_t0_gamma_forecasts_tail_quantiles():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = T0(
        repo_id="theforecastingcompany/t0-gamma",
        alias="t0-gamma",
        context_length=256,
        batch_size=2,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.01, 0.1, 0.9, 0.99])
    assert (fcst["t0-gamma-q-1"] < fcst["t0-gamma-q-10"]).all()
    assert (fcst["t0-gamma-q-99"] > fcst["t0-gamma-q-90"]).all()


def test_t0_gamma_bounds_forecasts_to_history():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    df["y"] = df["y"].abs().round()
    model = T0(
        repo_id="theforecastingcompany/t0-gamma",
        alias="t0-gamma",
        context_length=256,
        batch_size=2,
        clip_nonnegative=True,
        snap_integers=True,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.01])
    values = fcst[["t0-gamma", "t0-gamma-q-1"]].to_numpy()
    assert (values >= 0).all()
    np.testing.assert_array_equal(values, values.round())
