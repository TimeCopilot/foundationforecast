from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from tests.helpers import generate_series
from foundationforecast.core.conformal import ConformalIntervals
from foundationforecast.models.kumo import Kumo


class _FakeKumo:
    """Repeats the last observed value and records what it was asked to forecast."""

    def __init__(self):
        self.calls = []

    def forecast(self, series, h):
        self.calls.append((series.copy(), h))
        last = np.array([row[~np.isnan(row)][-1] for row in series])
        return np.repeat(last[:, None], h, axis=1).astype(np.float32)


def _patch_model(mocker, model: Kumo) -> _FakeKumo:
    fake = _FakeKumo()

    @contextmanager
    def _get_model():
        yield fake

    mocker.patch.object(model, "_get_model", _get_model)
    return fake


def test_kumo_defaults():
    model = Kumo()
    assert model.alias == "Kumo"
    assert model.repo_id == "nvidia/Kumo-Forecast"
    assert model.checkpoint == "run8_best_model_cr.pt"
    assert not model.supports_native_quantiles()
    assert not model.supports_native_futr_exog()


def test_kumo_invalid_context_length():
    with pytest.raises(ValueError, match="context_length"):
        Kumo(context_length=0)


def test_kumo_loads_the_requested_checkpoint(mocker):
    load = mocker.patch("foundationforecast.models.kumo.load_kumo_forecast")
    model = Kumo(repo_id="some/repo", checkpoint="other.pt")
    model._load_model()
    load.assert_called_once()
    assert load.call_args.args == ("some/repo",)
    assert load.call_args.kwargs["checkpoint"] == "other.pt"


def test_kumo_point_forecast_shape_and_dates(mocker):
    model = Kumo(batch_size=2)
    fake = _patch_model(mocker, model)
    df = generate_series(3, freq="D", min_length=20, max_length=20, equal_ends=True)
    h = 5
    fcst = model.forecast(df, h=h, freq="D")
    assert fcst.shape[0] == 3 * h
    assert list(fcst.columns) == ["unique_id", "ds", "Kumo"]
    last_y = df.groupby("unique_id")["y"].last()
    expected = fcst["unique_id"].map(last_y)
    np.testing.assert_allclose(fcst["Kumo"].to_numpy(), expected.to_numpy(), rtol=1e-5)
    # 3 series with batch_size=2 -> two batches, horizon passed through
    assert [call[1] for call in fake.calls] == [h, h]


def test_kumo_left_pads_short_series_with_nan(mocker):
    model = Kumo(batch_size=2)
    fake = _patch_model(mocker, model)
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 10 + ["b"] * 20,
            "ds": list(pd.date_range("2020-01-01", periods=10, freq="D"))
            + list(pd.date_range("2020-01-01", periods=20, freq="D")),
            "y": np.arange(30, dtype=float) + 1.0,
        }
    )
    model.forecast(df, h=2, freq="D")
    context, _ = fake.calls[0]
    assert context.shape == (2, 20)
    assert np.isnan(context[0, :10]).all()
    assert not np.isnan(context[1]).any()


def test_kumo_context_length_truncates_history(mocker):
    model = Kumo(context_length=8, batch_size=2)
    fake = _patch_model(mocker, model)
    df = generate_series(2, freq="D", min_length=30, max_length=30, equal_ends=True)
    model.forecast(df, h=2, freq="D")
    assert fake.calls[0][0].shape == (2, 8)


@pytest.mark.parametrize(
    "kwargs", [{"level": [80]}, {"quantiles": [0.1, 0.9]}], ids=["level", "quantiles"]
)
def test_kumo_intervals_require_conformal(mocker, kwargs):
    model = Kumo()
    _patch_model(mocker, model)
    df = generate_series(2, freq="D", min_length=40, max_length=40)
    with pytest.raises(ValueError, match="ConformalIntervals"):
        model.forecast(df, h=3, freq="D", **kwargs)
    with pytest.raises(ValueError, match="ConformalIntervals"):
        model.cross_validation(df, h=3, freq="D", n_windows=1, **kwargs)


def test_kumo_conformal_intervals_work(mocker):
    model = Kumo(
        batch_size=2,
        alias="Kumo-CF",
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    _patch_model(mocker, model)
    df = generate_series(2, freq="D", min_length=40, max_length=40)
    fcst = model.forecast(df, h=3, freq="D", level=[80])
    assert {"Kumo-CF", "Kumo-CF-lo-80", "Kumo-CF-hi-80"} <= set(fcst.columns)
    assert (fcst["Kumo-CF-lo-80"] <= fcst["Kumo-CF"]).all()
    assert (fcst["Kumo-CF"] <= fcst["Kumo-CF-hi-80"]).all()
    point = model.forecast(df, h=3, freq="D")
    assert list(point.columns) == ["unique_id", "ds", "Kumo-CF"]
