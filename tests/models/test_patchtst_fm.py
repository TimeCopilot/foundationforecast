import sys
from unittest.mock import MagicMock

import pytest
import torch

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    pytest.skip(
        "PatchTST-FM requires Python >= 3.11 and < 3.14",
        allow_module_level=True,
    )

from tests.helpers import generate_series  # noqa: E402
from foundationforecast.core.quantiles import (  # noqa: E402
    PATCHTST_FM_NATIVE_QUANTILES,
    quantile_column_name,
)
from foundationforecast.models.patchtst_fm import PatchTSTFM  # noqa: E402

pytestmark = pytest.mark.models

PATCHTST_FIXTURES = [
    pytest.param(
        {"context_length": 512, "batch_size": 2},
        id="patchtst-r1",
    ),
    pytest.param(
        {
            "repo_id": "ibm-granite/granite-timeseries-patchtst-fm-r2",
            "alias": "Granite-PatchTST-FM-r2",
            "context_length": 512,
            "batch_size": 2,
        },
        id="patchtst-r2",
    ),
]


def _patchtst(**kwargs) -> PatchTSTFM:
    return PatchTSTFM(**kwargs)


@pytest.mark.parametrize("model_kwargs", PATCHTST_FIXTURES)
def test_patchtst_forecast_uses_full_native_quantile_grid(mocker, model_kwargs):
    model = _patchtst(**model_kwargs)
    df = generate_series(n_series=2, freq="D", min_length=64, max_length=64)
    h = 3
    captured: list[list[float]] = []

    def fake_run(_hf_model, targets, horizon, quantile_levels):
        captured.append(list(quantile_levels))
        n_q = len(quantile_levels)
        return [torch.zeros(n_q, horizon).unsqueeze(-1) for _ in targets]

    mock_hf = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_hf
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)
    mocker.patch.object(model, "_run_model_on_targets", side_effect=fake_run)

    fcst = model.forecast(df, h=h, freq="D", level=[95])
    assert captured
    assert captured[0] == PATCHTST_FM_NATIVE_QUANTILES
    assert f"{model.alias}-lo-95" in fcst.columns
    assert f"{model.alias}-hi-95" in fcst.columns
    assert fcst[f"{model.alias}-lo-95"].le(fcst[f"{model.alias}-hi-95"]).all()


@pytest.mark.parametrize("model_kwargs", PATCHTST_FIXTURES)
@pytest.mark.parametrize(
    "quantiles",
    [
        [0.025, 0.975],
        [0.02, 0.5, 0.98],
    ],
)
def test_patchtst_forecast_interpolates_off_grid_quantiles(
    mocker, model_kwargs, quantiles
):
    model = _patchtst(**model_kwargs)
    df = generate_series(n_series=1, freq="D", min_length=32, max_length=32)
    h = 2
    n_knots = len(PATCHTST_FM_NATIVE_QUANTILES)

    def fake_run(_hf_model, targets, horizon, quantile_levels):
        assert list(quantile_levels) == PATCHTST_FM_NATIVE_QUANTILES
        # Monotone along quantile axis so interpolation stays ordered.
        ramp = torch.linspace(0.0, 1.0, n_knots)
        return [ramp.unsqueeze(-1).expand(-1, horizon).unsqueeze(-1) for _ in targets]

    mock_hf = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_hf
    mock_cm.__exit__.return_value = None
    mocker.patch.object(model, "_get_model", return_value=mock_cm)
    mocker.patch.object(model, "_run_model_on_targets", side_effect=fake_run)

    fcst = model.forecast(df, h=h, freq="D", quantiles=quantiles)
    exp_q_cols = [quantile_column_name(model.alias, q) for q in quantiles]
    assert all(col in fcst.columns for col in exp_q_cols)


@pytest.mark.parametrize("model_kwargs", PATCHTST_FIXTURES)
@pytest.mark.parametrize("level", [[95], [80, 95]])
def test_patchtst_using_level_including_95(model_kwargs, level):
    model = _patchtst(**model_kwargs)
    df = generate_series(n_series=2, freq="D", min_length=64, max_length=64)
    fcst_df = model.forecast(df, h=2, freq="D", level=level)
    exp_lv_cols = []
    for lv in level:
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    for lv in level:
        lo = f"{model.alias}-lo-{lv}"
        hi = f"{model.alias}-hi-{lv}"
        assert fcst_df[lo].le(fcst_df[hi]).mean() >= 0.8
