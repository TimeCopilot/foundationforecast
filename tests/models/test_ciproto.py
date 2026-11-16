import sys
import types
from contextlib import contextmanager

import pytest
import torch

from tests.helpers import generate_panel_with_futr_exog, generate_series
from foundationforecast.models.ciproto import (
    _CI_INSTALL_MSG,
    CIProto,
    _import_ci,
    is_ciproto_available,
)


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


def test_import_ci_raises_install_hint_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "continuousintelligence", None)
    with pytest.raises(ImportError, match=r"foundationforecast\[ci-proto\]"):
        _import_ci()


def test_import_ci_returns_module_when_installed(monkeypatch):
    fake_module = types.ModuleType("continuousintelligence")
    monkeypatch.setitem(sys.modules, "continuousintelligence", fake_module)
    assert _import_ci() is fake_module


def test_is_ciproto_available_reflects_installed_package(mocker):
    find_spec = mocker.patch("importlib.util.find_spec")
    find_spec.return_value = object()
    assert is_ciproto_available()
    find_spec.return_value = None
    assert not is_ciproto_available()


def test_ciproto_load_model_forwards_revision_and_device(mocker):
    ci = mocker.MagicMock()
    mocker.patch("foundationforecast.models.ciproto._import_ci", return_value=ci)
    model = CIProto(repo_id="org/ci", revision="abc123", device="cpu")
    loaded = model._load_model()
    ci.load_model.assert_called_once_with("org/ci", device="cpu", revision="abc123")
    assert loaded is ci.load_model.return_value


def test_ciproto_load_model_omits_revision_by_default(mocker):
    ci = mocker.MagicMock()
    mocker.patch("foundationforecast.models.ciproto._import_ci", return_value=ci)
    CIProto(device="cpu")._load_model()
    ci.load_model.assert_called_once_with(
        "ContinuousIntelligence/ci-proto", device="cpu"
    )


def test_ciproto_cache_prefix_depends_on_revision_and_device():
    default = CIProto()._model_cache_prefix()
    pinned = CIProto(revision="abc", device="cpu")._model_cache_prefix()
    assert default == "ciproto:ContinuousIntelligence/ci-proto:main:auto"
    assert pinned == "ciproto:ContinuousIntelligence/ci-proto:abc:cpu"


def test_ciproto_get_model_yields_loaded_model(mocker):
    model = CIProto(reuse_loaded_model=False)
    sentinel = object()
    mocker.patch.object(model, "_load_model", return_value=sentinel)
    with model._get_model() as loaded:
        assert loaded is sentinel


def test_ciproto_multiple_batches_and_left_padding(mocker):
    model = CIProto(batch_size=2, context_length=None)
    fake = _patch_model(mocker, model)
    df = generate_series(3, freq="D", min_length=20, max_length=40, seed=1)
    h = 4
    out = model.forecast(df, h=h)
    assert len(fake.calls) == 2  # 3 series with batch_size=2
    assert len(out) == 3 * h
    first = fake.calls[0]["history"]
    # shorter series are left-padded with NaN so rows align on the forecast origin
    assert first.shape[0] == 2
    assert torch.isnan(first).any() or first.shape[1] == 40


def test_ciproto_passes_timestamps_when_freq_known(mocker):
    model = CIProto(batch_size=4, context_length=30)
    fake = _patch_model(mocker, model)
    df = generate_series(2, freq="D", min_length=40, max_length=40, equal_ends=True)
    h = 5
    model.forecast(df, h=h, freq="D")
    call = fake.calls[0]
    assert call["freq"] == "D"
    assert call["timestamps"].shape == (2, 30 + h)
    assert call["timestamps"].dtype == torch.int64
    assert "future_covariates" not in call


def test_ciproto_native_future_covariates(mocker):
    model = CIProto(batch_size=4, context_length=20)
    fake = _patch_model(mocker, model)
    h = 3
    df, X_df, futr_exog_list = generate_panel_with_futr_exog(
        2, freq="D", h=h, min_length=32, max_length=32
    )
    out = model.forecast(df, h=h, freq="D", X_df=X_df, futr_exog_list=futr_exog_list)
    call = fake.calls[0]
    # [series, covariates, context + horizon]
    assert call["future_covariates"].shape == (2, len(futr_exog_list), 20 + h)
    assert len(out) == 2 * h
    # horizon values come from X_df, history values from df
    expected_future = torch.tensor(
        X_df[X_df["unique_id"] == call["series_ids"][0]][futr_exog_list]
        .to_numpy(dtype="float32")
        .T
    )
    assert torch.allclose(call["future_covariates"][0, :, -h:], expected_future)


def test_ciproto_level_intervals(mocker):
    model = CIProto(batch_size=4, context_length=40)
    _patch_model(mocker, model)
    df = generate_series(1, freq="h", min_length=50, max_length=50)
    out = model.forecast(df, h=4, level=[80])
    assert {"CIProto-lo-80", "CIProto-hi-80"} <= set(out.columns)
    assert (out["CIProto-lo-80"] <= out["CIProto-hi-80"]).all()


def test_ciproto_quantiles(mocker):
    model = CIProto(batch_size=4, context_length=40)
    _patch_model(mocker, model)
    df = generate_series(1, freq="h", min_length=50, max_length=50)
    out = model.forecast(df, h=4, quantiles=[0.1, 0.9])
    assert "CIProto-q-10" in out.columns
    assert "CIProto-q-90" in out.columns
    assert "CIProto" in out.columns
