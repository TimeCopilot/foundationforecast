from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.helpers import generate_series
from foundationforecast.models.tabby import TABBY_QUANTILES, Tabby

pytestmark = pytest.mark.models


def test_tabby_default_alias_and_repo():
    model = Tabby()
    assert model.alias == "Tabby"
    assert model.repo_id == "paris-noah/Tabby"
    assert not model.supports_native_futr_exog()


def test_tabby_predict_axis_maps_to_ff_columns(mocker):
    model = Tabby(context_length=64, batch_size=2)
    df = generate_series(2, freq="D", min_length=32, max_length=32, equal_ends=True)
    h = 4
    n_q = len(TABBY_QUANTILES)
    # Rising knots so 0.1 < 0.9 after remap from [B, 99, H]
    knots = np.linspace(0.0, 1.0, n_q, dtype=np.float32)
    fake = np.broadcast_to(knots[None, :, None], (2, n_q, h)).copy()

    mock_model = MagicMock()
    mock_model.quantile_levels = TABBY_QUANTILES
    mock_model.predict.return_value = _Tensor(fake)
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_model
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)

    fcst = model.forecast(df, h=h, freq="D", quantiles=[0.1, 0.5, 0.9])
    assert fcst.shape[0] == 2 * h
    assert "Tabby" in fcst.columns
    assert (fcst["Tabby-q-10"] < fcst["Tabby-q-90"]).all()
    assert mock_model.predict.call_args.kwargs["prediction_length"] == h


class _Tensor:
    def __init__(self, array: np.ndarray):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def test_tabby_forecasts_tail_quantiles():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = Tabby(context_length=256, batch_size=2)
    fcst = model.forecast(df, h=3, freq="D", quantiles=[0.01, 0.1, 0.9, 0.99])
    assert "Tabby" in fcst.columns
    assert (fcst["Tabby-q-1"] < fcst["Tabby-q-10"]).all()
    assert (fcst["Tabby-q-99"] > fcst["Tabby-q-90"]).all()


def test_tabby_level_columns():
    df = generate_series(1, freq="D", min_length=40, max_length=40)
    model = Tabby(context_length=256, batch_size=1)
    fcst = model.forecast(df, h=4, freq="D", level=[90])
    assert "Tabby-lo-90" in fcst.columns
    assert "Tabby-hi-90" in fcst.columns
    assert (fcst["Tabby-lo-90"] <= fcst["Tabby"]).all()
    assert (fcst["Tabby"] <= fcst["Tabby-hi-90"]).all()
