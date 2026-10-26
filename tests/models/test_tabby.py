from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from tests.helpers import generate_series
from foundationforecast.models.tabby import TABBY_QUANTILES, Tabby

pytestmark = pytest.mark.models


def _mock_model(mocker, model: Tabby, h: int) -> MagicMock:
    n_q = len(TABBY_QUANTILES)
    # Rising knots so 0.1 < 0.9 after remap from [B, 99, H]
    knots = torch.linspace(0.0, 1.0, n_q)
    backend = MagicMock()
    backend.quantile_levels = TABBY_QUANTILES
    backend.side_effect = lambda **kwargs: {
        "quantile_preds": knots[None, :, None].expand(
            kwargs["context"].shape[0], n_q, h
        )
    }
    cm = MagicMock()
    cm.__enter__.return_value = backend
    cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=cm)
    return backend


def test_tabby_default_alias_and_repo():
    model = Tabby()
    assert model.alias == "Tabby"
    assert model.repo_id == "paris-noah/Tabby"
    assert model.precision == "bf16"
    assert not model.supports_native_futr_exog()


def test_tabby_predict_axis_maps_to_ff_columns(mocker):
    model = Tabby(context_length=64, batch_size=2)
    df = generate_series(2, freq="D", min_length=32, max_length=32, equal_ends=True)
    h = 4
    backend = _mock_model(mocker, model, h)

    fcst = model.forecast(df, h=h, freq="D", quantiles=[0.1, 0.5, 0.9])
    assert fcst.shape[0] == 2 * h
    assert "Tabby" in fcst.columns
    assert (fcst["Tabby-q-10"] < fcst["Tabby-q-90"]).all()
    assert backend.call_args.kwargs["prediction_length"] == h


def test_tabby_flags_left_padding_and_keeps_fp32_context(mocker):
    model = Tabby(context_length=16, batch_size=2)
    backend = _mock_model(mocker, model, h=2)
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 10 + ["b"] * 20,
            "ds": list(pd.date_range("2020-01-01", periods=10, freq="D"))
            + list(pd.date_range("2020-01-01", periods=20, freq="D")),
            "y": np.r_[np.full(10, 12345.678), np.arange(20) + 0.001],
        }
    )
    df.loc[25, "y"] = np.nan

    model.forecast(df, h=2, freq="D")

    kwargs = backend.call_args.kwargs
    context = kwargs["context"]
    mask = kwargs["context_mask"]
    pad = kwargs["past_is_pad"]
    assert context.shape == (2, 16)
    assert context.dtype == torch.float32
    assert pad[0].tolist() == [True] * 6 + [False] * 10
    assert not pad[1].any()
    np.testing.assert_allclose(context[0, 6:].numpy(), 12345.678, rtol=1e-7)
    assert mask[0, :6].sum() == 0
    assert mask[1].sum() == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast needs CUDA")
@pytest.mark.parametrize(
    ("precision", "dtype"), [("bf16", torch.bfloat16), ("fp32", None)]
)
def test_tabby_precision_controls_autocast(mocker, precision, dtype):
    model = Tabby(batch_size=2, precision=precision)
    backend = _mock_model(mocker, model, h=2)
    inner = backend.side_effect
    seen = []

    def record(**kwargs):
        enabled = torch.is_autocast_enabled("cuda")
        seen.append(torch.get_autocast_dtype("cuda") if enabled else None)
        return inner(**kwargs)

    backend.side_effect = record
    df = generate_series(1, freq="D", min_length=20, max_length=20)
    model.forecast(df, h=2, freq="D")
    assert seen == [dtype]


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
