from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tests.helpers import generate_series
from foundationforecast.models.simons import Simons

pytestmark = pytest.mark.models


def test_simons_native_futr_exog_passes_covariates(mocker):
    model = Simons(context_length=64, batch_size=2, exog_strategy="auto")
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
    mock_forecaster.forecast.return_value = np.broadcast_to(
        np.linspace(0.1, 0.9, 9),
        (1, h, 9),
    ).copy()

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_forecaster
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)

    fcst = model.forecast(df, h=h, freq="D", X_df=X_df)
    assert fcst.shape[0] == h
    kwargs = mock_forecaster.forecast.call_args.kwargs
    history = mock_forecaster.forecast.call_args.args[0]
    assert history.shape == (1, min(n, 64))
    assert kwargs["horizon"] == h
    assert kwargs["covariates"].shape == (1, 1, min(n, 64) + h)
    assert fcst["Simons"].notna().all()


def test_load_model_passes_repo_and_device(mocker):
    model = Simons(repo_id="/tmp/simons-local")
    mocked = mocker.patch(
        "foundationforecast.models.simons.SimonsModel.from_pretrained",
        return_value=MagicMock(),
    )
    loaded = model._load_model()
    mocked.assert_called_once_with("/tmp/simons-local", device=model.device)
    assert loaded is mocked.return_value


def test_simons_forecasts_native_quantiles():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = Simons(
        repo_id="abel-lab/Simons-25M",
        alias="Simons-25M",
        context_length=256,
        batch_size=2,
    )
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.1, 0.5, 0.9])
    assert (fcst["Simons-25M-q-10"] < fcst["Simons-25M"]).all()
    assert (fcst["Simons-25M"] < fcst["Simons-25M-q-90"]).all()
