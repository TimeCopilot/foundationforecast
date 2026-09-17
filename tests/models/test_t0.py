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

from foundationforecast.models.t0 import T0  # noqa: E402
from tests.helpers import generate_series  # noqa: E402

pytestmark = pytest.mark.models


def test_t0_forecast_passes_quantile_levels(mocker):
    model = T0(context_length=64, batch_size=2)
    df = generate_series(2, freq="D", min_length=32, max_length=32)

    mock_forecaster = MagicMock()
    mock_out = MagicMock()
    mock_out.quantiles.cpu.return_value.numpy.return_value = np.ones((2, 3, 1))
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
