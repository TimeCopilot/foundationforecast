from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tests.helpers import generate_series
from foundationforecast.models.timestral import Timestral

pytestmark = pytest.mark.models


def test_timestral_native_futr_exog_passes_known_covariates(mocker):
    model = Timestral(context_length=64, batch_size=2, exog_strategy="auto")
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
    levels = tuple(np.linspace(0.01, 0.99, 21))
    mock_out = MagicMock()
    mock_out.quantile_levels = levels
    mock_out.quantiles = np.ones((1, 1, h, 21))
    mock_out.median = np.ones((1, 1, h))
    mock_forecaster.forecast.return_value = mock_out

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_forecaster
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)

    fcst = model.forecast(df, h=h, freq="D", X_df=X_df)
    assert fcst.shape[0] == h
    kwargs = mock_forecaster.forecast.call_args.kwargs
    assert kwargs["known_covariates"] is not None
    assert kwargs["known_covariates"].shape == (1, 1, min(n, 64) + h)
    assert kwargs["symmetric_averaging"] is True
    assert kwargs["clip_nonnegative"] is True


def test_timestral_forecasts_tail_quantiles():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = Timestral(
        repo_id="mistralai/Timestral-5M",
        alias="Timestral-5M",
        context_length=256,
        batch_size=2,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.01, 0.1, 0.9, 0.99])
    assert (fcst["Timestral-5M-q-1"] < fcst["Timestral-5M-q-10"]).all()
    assert (fcst["Timestral-5M-q-99"] > fcst["Timestral-5M-q-90"]).all()


def test_timestral_clips_nonnegative_forecasts():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    df["y"] = df["y"].abs()
    model = Timestral(
        repo_id="mistralai/Timestral-5M",
        alias="Timestral-5M",
        context_length=256,
        batch_size=2,
        clip_nonnegative=True,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.01])
    values = fcst[["Timestral-5M", "Timestral-5M-q-1"]].to_numpy()
    assert (values >= 0).all()
