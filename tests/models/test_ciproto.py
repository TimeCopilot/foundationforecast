from contextlib import contextmanager

import pytest
import torch

from tests.helpers import generate_series
from foundationforecast.models.ciproto import _CI_INSTALL_MSG, CIProto


class _FakeForecaster:
    def __init__(self):
        self.calls: list[dict] = []
        self.quantiles = torch.linspace(0.1, 0.9, 9)

    def forecast(self, history, **kwargs):
        self.calls.append({"history": history, **kwargs})
        v, t = history.shape
        h = kwargs["horizon"]
        q = len(self.quantiles)
        base = torch.nanmedian(history, dim=1).values
        pred = base[:, None, None].expand(v, q, h).clone()
        return pred.cpu().float()


def _patch_model(mocker, model: CIProto) -> _FakeForecaster:
    fake = _FakeForecaster()

    @contextmanager
    def _get_model():
        yield fake

    mocker.patch.object(model, "_get_model", _get_model)
    return fake


def test_ciproto_defaults():
    model = CIProto()
    assert model.alias == "CIProto"
    assert model.repo_id == "ContinuousIntelligence/ci-proto"
    assert model.supports_native_futr_exog()
    assert model.supports_native_quantiles()


def test_ciproto_invalid_context_length():
    with pytest.raises(ValueError, match="context_length"):
        CIProto(context_length=0)


def test_ciproto_load_requires_extra(mocker):
    mocker.patch(
        "foundationforecast.models.ciproto._import_ci",
        side_effect=ImportError(_CI_INSTALL_MSG),
    )
    model = CIProto()
    with pytest.raises(ImportError, match="ci-proto"):
        model._load_model()


def test_ciproto_batch_forecast_shapes(mocker):
    model = CIProto(batch_size=2, context_length=50)
    fake = _patch_model(mocker, model)
    h = 6
    df = generate_series(
        2,
        freq="D",
        min_length=80,
        max_length=80,
        equal_ends=True,
    )
    out = model.forecast(df, h=h)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert len(call["series_ids"]) == 2
    assert call["history"].shape == (2, 50)
    assert list(out.columns) == ["unique_id", "ds", "CIProto"]
    assert len(out) == 2 * h


def test_ciproto_quantiles(mocker):
    model = CIProto(batch_size=4, context_length=40)
    _patch_model(mocker, model)
    df = generate_series(1, freq="h", min_length=50, max_length=50)
    out = model.forecast(df, h=4, quantiles=[0.1, 0.9])
    assert "CIProto-q-10" in out.columns
    assert "CIProto-q-90" in out.columns
    assert "CIProto" in out.columns
