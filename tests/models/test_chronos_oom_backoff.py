import pytest
import torch

from foundationforecast.models.chronos import _QuantileBatchPredictor


class _FakePipeline:
    """Raises CUDA OOM while ``batch_size`` is above ``fits_at``."""

    def __init__(self, fits_at: int | None = None, fail_times: int = 0) -> None:
        self.fits_at = fits_at
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def predict_quantiles(self, batch, **kwargs):
        self.calls.append(kwargs)
        if self.fits_at is not None and kwargs.get("batch_size", 0) > self.fits_at:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return ("quantiles", "mean", batch)


def _predictor(model, *, batch_size: int, adaptive: bool) -> _QuantileBatchPredictor:
    return _QuantileBatchPredictor(
        model,
        prediction_length=3,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=batch_size,
        adaptive_batch_size=adaptive,
    )


def test_adaptive_halves_batch_size_until_it_fits_and_keeps_it():
    model = _FakePipeline(fits_at=4)
    predict = _predictor(model, batch_size=16, adaptive=True)

    assert predict("b0") == ("quantiles", "mean", "b0")
    assert [c["batch_size"] for c in model.calls] == [16, 8, 4]
    assert predict.batch_size == 4

    # Later batches start from the reduced size without retrying.
    assert predict("b1") == ("quantiles", "mean", "b1")
    assert model.calls[-1]["batch_size"] == 4
    assert len(model.calls) == 4


def test_adaptive_passes_prediction_kwargs():
    model = _FakePipeline()
    predict = _predictor(model, batch_size=8, adaptive=True)
    predict("b0")
    assert model.calls == [
        {"prediction_length": 3, "quantile_levels": [0.1, 0.5, 0.9], "batch_size": 8}
    ]


def test_adaptive_reraises_when_batch_size_cannot_be_halved():
    model = _FakePipeline(fits_at=0)
    predict = _predictor(model, batch_size=2, adaptive=True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict("b0")
    # 2 → 1, then OOM at 1 is fatal.
    assert [c["batch_size"] for c in model.calls] == [2, 1]
    assert predict.batch_size == 1


def test_non_adaptive_pipeline_reraises_and_does_not_pass_batch_size():
    model = _FakePipeline(fail_times=1)
    predict = _predictor(model, batch_size=16, adaptive=False)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict("b0")
    assert len(model.calls) == 1
    assert "batch_size" not in model.calls[0]
    assert predict.batch_size == 16
