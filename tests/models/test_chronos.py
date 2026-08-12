import pytest
import torch

from tests.helpers import generate_series
from foundationforecast.models.chronos import Chronos, ChronosFinetuningConfig

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
