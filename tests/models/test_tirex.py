import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "TiRex requires Python >= 3.11",
        allow_module_level=True,
    )

from tests.helpers import generate_series  # noqa: E402
from foundationforecast.models.tirex import TiRex  # noqa: E402

pytestmark = pytest.mark.models


def test_is_tirex2_dispatch():
    assert not TiRex(repo_id="NX-AI/TiRex").supports_native_futr_exog()
    assert not TiRex(repo_id="NX-AI/TiRex")._is_tirex2()
    assert not TiRex(repo_id="NX-AI/TiRex-1.1-gifteval")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2").supports_native_futr_exog()
    assert TiRex(repo_id="NX-AI/TiRex-2/")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2-gifteval-pretrain")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2-gifteval-zs")._is_tirex2()


def test_tirex2_forecast():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = TiRex(repo_id="NX-AI/TiRex-2", alias="TiRex-2", batch_size=2)
    fcst = model.forecast(df, h=3, freq="D")
    assert fcst.shape == (6, 3)
    assert "TiRex-2" in fcst.columns


def test_tirex2_native_futr_exog_passes_future_covariates(mocker):
    model = TiRex(repo_id="NX-AI/TiRex-2", batch_size=1)
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

    mock_model = MagicMock()
    mock_model.forecast.return_value = [np.ones((1, 9, h), dtype=np.float32)]

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_model
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model_v2", return_value=mock_cm)

    fcst = model.forecast(df, h=h, freq="D", X_df=X_df)
    assert fcst.shape[0] == h
    timeseries = mock_model.forecast.call_args.kwargs["timeseries"]
    assert timeseries[0].future_covariates is not None
    assert timeseries[0].future_covariates.shape == (1, n + h)
