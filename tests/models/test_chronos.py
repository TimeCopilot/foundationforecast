import pytest
import torch

from tests.helpers import generate_series
from foundationforecast.models.chronos import (
    Chronos,
    ChronosFinetuningConfig,
    _QuantileBatchPredictor,
)

pytestmark = pytest.mark.models


def test_chronos_default_dtype_is_float32():
    """Ensure Chronos defaults to float32 dtype."""
    model = Chronos(repo_id="amazon/chronos-t5-tiny")
    assert model.dtype == torch.float32


def test_chronos_forecast_with_bfloat16():
    """Ensure Chronos runs a real forecast with a custom dtype."""
    model = Chronos(
        repo_id="amazon/chronos-bolt-tiny",
        dtype=torch.bfloat16,
        alias="Chronos-Bolt",
    )
    df = generate_series(n_series=1, freq="D", min_length=20, max_length=20)
    fcst = model.forecast(df=df, h=2, freq="D")
    assert fcst.shape == (2, 3)
    assert "Chronos-Bolt" in fcst.columns


def test_chronos_finetuning_save_and_reuse(tmp_path):
    """Finetune with save_path, run cross-validation,
    then forecast using the saved path."""
    save_path = tmp_path / "chronos2-finetuned"
    config = ChronosFinetuningConfig(
        finetune_steps=2,
        save_path=save_path,
    )
    model = Chronos(
        repo_id="autogluon/chronos-2-small",
        finetuning_config=config,
        batch_size=2,
    )
    n_series = 2
    df = generate_series(n_series, freq="MS")

    cv_df = model.cross_validation(df, h=2, n_windows=1, freq="MS")
    assert not cv_df.empty
    assert "Chronos" in cv_df.columns

    assert save_path.is_dir(), f"Finetuned model should be saved to {save_path}"
    assert (save_path / "config.json").exists(), "Expected config.json "

    model_reuse = Chronos(
        repo_id=str(save_path),
        finetuning_config=None,
        batch_size=2,
    )
    fcst = model_reuse.forecast(df, h=2, freq="MS")
    assert not fcst.empty
    assert "Chronos" in fcst.columns
    assert len(fcst) == n_series * 2  # h=2 per series


def test_chronos_lora_finetuning_save_and_reuse(tmp_path):
    """Finetune Chronos-2 with LoRA and save_path, then load from path and forecast."""
    pytest.importorskip("peft")

    save_path = tmp_path / "chronos2-lora-finetuned"
    config = ChronosFinetuningConfig(
        finetune_steps=2,
        finetune_mode="lora",
        learning_rate=1e-5,
        save_path=save_path,
    )
    model = Chronos(
        repo_id="autogluon/chronos-2-small",
        finetuning_config=config,
        batch_size=2,
    )
    n_series = 2
    df = generate_series(n_series, freq="MS")

    fcst = model.forecast(df, h=2, freq="MS")
    assert not fcst.empty
    assert "Chronos" in fcst.columns

    assert save_path.is_dir(), f"LoRA checkpoint should be saved to {save_path}"
    assert (save_path / "adapter_config.json").exists(), "Expected adapter_config.json "

    model_reuse = Chronos(
        repo_id=str(save_path),
        finetuning_config=None,
        batch_size=2,
    )
    fcst_reuse = model_reuse.forecast(df, h=2, freq="MS")
    assert not fcst_reuse.empty
    assert "Chronos" in fcst_reuse.columns
    assert len(fcst_reuse) == n_series * 2


class _FakeOOMPipeline:
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


def _oom_predictor(model, *, batch_size: int, adaptive: bool):
    return _QuantileBatchPredictor(
        model,
        prediction_length=3,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=batch_size,
        adaptive_batch_size=adaptive,
    )


def test_oom_backoff_halves_batch_size_until_it_fits_and_keeps_it():
    model = _FakeOOMPipeline(fits_at=4)
    predict = _oom_predictor(model, batch_size=16, adaptive=True)

    assert predict("b0") == ("quantiles", "mean", "b0")
    assert [c["batch_size"] for c in model.calls] == [16, 8, 4]
    assert predict.batch_size == 4

    # Later batches start from the reduced size without retrying.
    assert predict("b1") == ("quantiles", "mean", "b1")
    assert model.calls[-1]["batch_size"] == 4
    assert len(model.calls) == 4


def test_oom_backoff_passes_prediction_kwargs():
    model = _FakeOOMPipeline()
    predict = _oom_predictor(model, batch_size=8, adaptive=True)
    predict("b0")
    assert model.calls == [
        {"prediction_length": 3, "quantile_levels": [0.1, 0.5, 0.9], "batch_size": 8}
    ]


def test_oom_backoff_reraises_when_batch_size_cannot_be_halved():
    model = _FakeOOMPipeline(fits_at=0)
    predict = _oom_predictor(model, batch_size=2, adaptive=True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict("b0")
    # 2 → 1, then OOM at 1 is fatal.
    assert [c["batch_size"] for c in model.calls] == [2, 1]
    assert predict.batch_size == 1


def test_oom_backoff_non_adaptive_pipeline_reraises_without_batch_size():
    model = _FakeOOMPipeline(fail_times=1)
    predict = _oom_predictor(model, batch_size=16, adaptive=False)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict("b0")
    assert len(model.calls) == 1
    assert "batch_size" not in model.calls[0]
    assert predict.batch_size == 16
