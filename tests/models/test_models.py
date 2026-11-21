import inspect
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from tests.helpers import (
    generate_panel_with_futr_exog,
    generate_series,
    panel_with_futr_exog_horizon,
)
from .conftest import conformal_models, models
from foundationforecast import ACI, PID
from foundationforecast.core.conformal import ConformalIntervals
from foundationforecast.core.quantiles import quantile_column_name


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
def test_freq_inferred_correctly(model, freq):
    n_series = 2
    df = generate_series(
        n_series,
        freq=freq,
    )
    fcsts_no_freq = model.forecast(df, h=3)
    fcsts_with_freq = model.forecast(df, h=3, freq=freq)
    cv_no_freq = model.cross_validation(df, h=3)
    cv_with_freq = model.cross_validation(df, h=3, freq=freq)
    # some foundation models produce different results
    # each time they are called
    cols_to_check = ["unique_id", "ds"]
    cols_to_check_cv = ["unique_id", "ds", "y", "cutoff"]
    pd.testing.assert_frame_equal(
        fcsts_no_freq[cols_to_check],
        fcsts_with_freq[cols_to_check],
    )
    pd.testing.assert_frame_equal(
        cv_no_freq[cols_to_check_cv],
        cv_with_freq[cols_to_check_cv],
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "freq",
    [
        # gift eval freqs
        "10S",
        "10T",
        "15T",
        "5T",
        "A-DEC",
        "D",
        "H",
        "M",
        "MS",
        "Q-DEC",
        "W-FRI",
        "W-SUN",
        "W-THU",
        "W-TUE",
        "W-WED",
    ],
)
@pytest.mark.parametrize("h", [1, 12])
def test_correct_forecast_dates(model, freq, h):
    n_series = 5
    df = generate_series(
        n_series,
        freq=freq,
        min_length=50,
        max_length=50,
    )
    df_test = df.groupby("unique_id").tail(h)
    df_train = df.drop(df_test.index)
    fcst_df = model.forecast(
        df_train,
        h=h,
        freq=freq,
    )
    exp_n_cols = 3
    assert fcst_df.shape == (n_series * h, exp_n_cols)
    exp_cols = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        fcst_df[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        df_test[exp_cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("freq", ["H", "D", "W-MON", "MS"])
@pytest.mark.parametrize("n_windows", [1, 4])
def test_cross_validation(model, freq, n_windows):
    h = 12
    n_series = 5
    df = generate_series(n_series, freq=freq, equal_ends=True)
    cv_df = model.cross_validation(
        df,
        h=h,
        freq=freq,
        n_windows=n_windows,
    )
    exp_n_cols = 5
    assert cv_df.shape == (n_series * h * n_windows, exp_n_cols)
    cutoffs = cv_df["cutoff"].unique()
    assert len(cutoffs) == n_windows
    df_test = df.groupby("unique_id").tail(h * n_windows)
    exp_cols = ["unique_id", "ds", "y"]
    pd.testing.assert_frame_equal(
        cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        df_test.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
    )
    if n_windows == 1:
        df_test = df.groupby("unique_id").tail(h)
        df_train = df.drop(df_test.index)
        fcst_df = model.forecast(
            df_train,
            h=h,
            freq=freq,
        )
        exp_cols = ["unique_id", "ds"]
        pd.testing.assert_frame_equal(
            cv_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
            fcst_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)[exp_cols],
        )


def _assert_quantile_monotonicity(model, fcst_df, ordered_q_cols):
    alias = model.alias.lower()
    for c1, c2 in zip(ordered_q_cols[:-1], ordered_q_cols[1:], strict=False):
        if "chronos" in alias or "median" in alias:
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif (
            "timesfm" in alias
            or "flowstate" in alias
            or "toto" in alias
            or "tafsut" in alias
        ):
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.8
        elif (
            "tirex" in alias
            or alias.startswith("t0")
            or "patchtst" in alias
            or "granite" in alias
        ):
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif "tabpfn" in alias:
            continue
        elif "moe" in alias:
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.5
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()


def _assert_level_monotonicity(model, fcst_df, exp_lv_cols):
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        if "chronos" in model.alias.lower() or "median" in model.alias.lower():
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif "tabpfn" in model.alias.lower():
            continue
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()


@pytest.mark.parametrize("model", models)
def test_exog_forecast_with_X_df(model):
    """Hub models with native futr exog accept df + X_df."""
    if not model.supports_native_futr_exog():
        pytest.skip("known-future exog requires native checkpoint support")
    h = 3
    n_series = 2
    df, X_df, futr_exog_list = generate_panel_with_futr_exog(
        n_series,
        freq="D",
        h=h,
        min_length=32,
        max_length=32,
    )
    fcst_exog = model.forecast(
        df=df,
        h=h,
        freq="D",
        X_df=X_df,
        futr_exog_list=futr_exog_list,
    )
    assert fcst_exog.shape[0] == n_series * h
    assert model.alias in fcst_exog.columns
    assert fcst_exog[model.alias].notna().all()
    exp_cols = ["unique_id", "ds"]
    pd.testing.assert_frame_equal(
        fcst_exog[exp_cols].sort_values(exp_cols).reset_index(drop=True),
        X_df[exp_cols].sort_values(exp_cols).reset_index(drop=True),
    )


@pytest.mark.parametrize("model", models)
def test_exog_using_quantiles(model):
    if not model.supports_native_futr_exog():
        pytest.skip("known-future exog requires native checkpoint support")
    h = 2
    n_series = 3
    df, X_df, futr_exog_list = generate_panel_with_futr_exog(
        n_series,
        freq="D",
        h=h,
        min_length=32,
        max_length=32,
    )
    qs = [round(i * 0.1, 1) for i in range(1, 10)] + [0.57]
    fcst_df = model.forecast(
        df=df,
        h=h,
        freq="D",
        quantiles=qs,
        X_df=X_df,
        futr_exog_list=futr_exog_list,
    )
    exp_qs_cols = [quantile_column_name(model.alias, q) for q in qs]
    assert len(exp_qs_cols) == len(fcst_df.columns) - 3
    assert all(col in fcst_df.columns for col in exp_qs_cols)
    legacy_057 = f"{model.alias}-q-{int(0.57 * 100)}"
    assert legacy_057 not in fcst_df.columns
    assert not any(("-lo-" in col or "-hi-" in col) for col in fcst_df.columns)
    ordered_q_cols = [
        quantile_column_name(model.alias, q) for q in sorted(qs, key=float)
    ]
    _assert_quantile_monotonicity(model, fcst_df, ordered_q_cols)


@pytest.mark.parametrize("model", models)
def test_exog_using_level(model):
    if not model.supports_native_futr_exog():
        pytest.skip("known-future exog requires native checkpoint support")
    h = 2
    n_series = 2
    df, X_df, futr_exog_list = generate_panel_with_futr_exog(
        n_series,
        freq="D",
        h=h,
        min_length=32,
        max_length=32,
    )
    level = [20, 40, 50, 60, 80]
    fcst_df = model.forecast(
        df=df,
        h=h,
        freq="D",
        level=level,
        X_df=X_df,
        futr_exog_list=futr_exog_list,
    )
    exp_lv_cols = []
    for lv in level:
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert len(exp_lv_cols) == len(fcst_df.columns) - 3
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df.columns)
    _assert_level_monotonicity(model, fcst_df, exp_lv_cols)


@pytest.mark.parametrize("model", models)
def test_exog_cross_validation_using_level(model):
    if not model.supports_native_futr_exog():
        pytest.skip("known-future exog requires native checkpoint support")
    h = 2
    n_series = 2
    df_hist, X_df, futr_exog_list = generate_panel_with_futr_exog(
        n_series,
        freq="D",
        h=h,
        min_length=32,
        max_length=32,
    )
    panel = panel_with_futr_exog_horizon(df_hist, X_df)
    level = [80, 90]
    cv_df = model.cross_validation(
        df=panel,
        h=h,
        freq="D",
        n_windows=1,
        level=level,
        futr_exog_list=futr_exog_list,
    )
    assert len(cv_df) == n_series * h
    for lv in level:
        assert f"{model.alias}-lo-{lv}" in cv_df.columns
        assert f"{model.alias}-hi-{lv}" in cv_df.columns


@pytest.mark.parametrize("model", models)
def test_passing_both_level_and_quantiles(model):
    df = generate_series(n_series=1, freq="D")
    with pytest.raises(ValueError):
        model.forecast(
            df=df,
            h=1,
            freq="D",
            level=[80, 95],
            quantiles=[0.1, 0.5, 0.9],
        )
    with pytest.raises(ValueError):
        model.cross_validation(
            df=df,
            h=1,
            freq="D",
            level=[80, 95],
            quantiles=[0.1, 0.5, 0.9],
        )


@pytest.mark.parametrize("model", models)
def test_using_quantiles(model):
    if not model.supports_native_quantiles():
        pytest.skip("no native quantiles; covered by the conformal tests")
    # 0.57: int(100×q) truncates to 56; output must use suffix 57 (100×q text).
    # 0.025 / 0.975: off 0.01 knot grid (interpolated on fixed-knot backends).
    qs = [round(i * 0.1, 1) for i in range(1, 10)] + [0.025, 0.57, 0.975]
    df = generate_series(n_series=3, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=2,
        freq="D",
        quantiles=qs,
    )
    exp_qs_cols = [quantile_column_name(model.alias, q) for q in qs]
    assert len(exp_qs_cols) == len(fcst_df.columns) - 3
    assert all(col in fcst_df.columns for col in exp_qs_cols)
    legacy_057 = f"{model.alias}-q-{int(0.57 * 100)}"
    assert legacy_057 not in fcst_df.columns
    assert not any(("-lo-" in col or "-hi-" in col) for col in fcst_df.columns)
    # test monotonicity of quantiles (sorted by q, not request order)
    ordered_q_cols = [
        quantile_column_name(model.alias, q) for q in sorted(qs, key=float)
    ]
    _assert_quantile_monotonicity(model, fcst_df, ordered_q_cols)


@pytest.mark.parametrize("model", models)
def test_using_level(model):
    if not model.supports_native_quantiles():
        pytest.skip("no native quantiles; covered by the conformal tests")
    # 20/40/60/80 hit decile knots; 50 → 0.25/0.75; 95 → 0.025/0.975 (interpolated)
    level = [20, 40, 50, 60, 80, 95]
    df = generate_series(n_series=2, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=2,
        freq="D",
        level=level,
    )
    exp_lv_cols = []
    for lv in level:
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert len(exp_lv_cols) == len(fcst_df.columns) - 3
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df.columns)
    _assert_level_monotonicity(model, fcst_df, exp_lv_cols)


@pytest.mark.models
@pytest.mark.parametrize("model", conformal_models)
def test_conformal_forecast_using_level(model):
    h = 3
    level = [80, 90]
    df = generate_series(n_series=2, freq="D", min_length=40, max_length=40)
    fcst_df = model.forecast(df=df, h=h, freq="D", level=level)
    exp_lv_cols = []
    for lv in level:
        exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert all(col in fcst_df.columns for col in exp_lv_cols)
    assert not any("-q-" in col for col in fcst_df.columns)
    _assert_level_monotonicity(model, fcst_df, exp_lv_cols)


@pytest.mark.models
@pytest.mark.parametrize("model", models)
def test_finite_sample_conformal_forecast(model, monkeypatch):
    monkeypatch.setattr(
        model,
        "prediction_intervals",
        ConformalIntervals(n_windows=4, method="conformal_error_finite_sample"),
    )
    df = generate_series(n_series=2, freq="D", min_length=40, max_length=40)
    with pytest.warns(
        UserWarning,
        match=(
            "level 95 needs at least 19 calibration windows, got 4; bounds are infinite"
        ),
    ) as caught:
        fcst = model.forecast(df=df, h=3, freq="D", level=[50, 80, 95])
    infinite_warnings = [
        warning for warning in caught if "bounds are infinite" in str(warning.message)
    ]
    assert len(infinite_warnings) == 1
    assert fcst.shape == (6, 9)
    assert np.isfinite(fcst[model.alias]).all()
    for level in [50, 80]:
        lo, hi = fcst[f"{model.alias}-lo-{level}"], fcst[f"{model.alias}-hi-{level}"]
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert lo.le(fcst[model.alias]).all()
        assert hi.ge(fcst[model.alias]).all()
    assert fcst[f"{model.alias}-lo-80"].le(fcst[f"{model.alias}-lo-50"]).all()
    assert fcst[f"{model.alias}-hi-80"].ge(fcst[f"{model.alias}-hi-50"]).all()
    assert np.isneginf(fcst[f"{model.alias}-lo-95"]).all()
    assert np.isposinf(fcst[f"{model.alias}-hi-95"]).all()


@pytest.mark.models
@pytest.mark.parametrize("model", models)
def test_pid_conformal_forecast(model, monkeypatch):
    monkeypatch.setattr(
        model, "prediction_intervals", ConformalIntervals(n_windows=6, method=PID())
    )
    level = [80, 95]
    df = generate_series(n_series=2, freq="D", min_length=60, max_length=60)
    with warnings.catch_warnings():
        # short backtests may saturate the integrator to infinite bounds
        warnings.filterwarnings("ignore", message="PID error integration saturated")
        fcst = model.forecast(df=df, h=3, freq="D", level=level)
    cols = [f"{model.alias}-{side}-{lv}" for lv in level for side in ("lo", "hi")]
    assert fcst.shape == (6, 3 + len(cols))
    assert not fcst[[model.alias, *cols]].isna().any().any()
    for lv in level:
        assert (fcst[f"{model.alias}-lo-{lv}"] <= fcst[model.alias]).all()
        assert (fcst[f"{model.alias}-hi-{lv}"] >= fcst[model.alias]).all()
    for lv in level:
        assert (fcst[f"{model.alias}-lo-{lv}"] <= fcst[f"{model.alias}-hi-{lv}"]).all()
    assert (fcst[f"{model.alias}-lo-95"] <= fcst[f"{model.alias}-lo-80"]).all()
    assert (fcst[f"{model.alias}-hi-95"] >= fcst[f"{model.alias}-hi-80"]).all()


@pytest.mark.models
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("use_quantiles", [False, True])
def test_aci_conformal_forecast(model, use_quantiles, monkeypatch):
    monkeypatch.setattr(
        model,
        "prediction_intervals",
        ConformalIntervals(n_windows=6, method=ACI(gamma=0.005)),
    )
    df = generate_series(n_series=2, freq="D", min_length=60, max_length=60)
    kwargs = (
        {"quantiles": [0.025, 0.1, 0.5, 0.9, 0.975]}
        if use_quantiles
        else {"level": [80, 95]}
    )
    forecast = model.forecast(df=df, h=3, freq="D", **kwargs)
    numeric = forecast.drop(columns=["unique_id", "ds"])
    assert len(forecast) == 6
    assert np.isfinite(numeric).all().all()
    if use_quantiles:
        columns = [f"{model.alias}-q-{q}" for q in ["2.5", "10", "50", "90", "97.5"]]
        assert set(forecast) == {"unique_id", "ds", model.alias, *columns}
        for low, high in zip(columns, columns[1:], strict=False):
            assert forecast[low].le(forecast[high]).all()
        np.testing.assert_array_equal(forecast[columns[2]], forecast[model.alias])
    else:
        assert forecast.shape == (6, 7)
        for level in [80, 95]:
            assert forecast[f"{model.alias}-lo-{level}"].le(forecast[model.alias]).all()
            assert forecast[f"{model.alias}-hi-{level}"].ge(forecast[model.alias]).all()
        assert (
            forecast[f"{model.alias}-lo-95"].le(forecast[f"{model.alias}-lo-80"]).all()
        )
        assert (
            forecast[f"{model.alias}-hi-95"].ge(forecast[f"{model.alias}-hi-80"]).all()
        )


def _local_repo_case_id(model: Any) -> str:
    repo_id = getattr(model, "repo_id", None)
    alias = getattr(model, "alias", type(model).__name__)
    batch = getattr(model, "batch_size", None)
    repo = repo_id.rsplit("/", 1)[-1] if isinstance(repo_id, str) else "no-repo"
    return f"{alias}-{repo}-bs{batch}"


def _public_class(model: Any) -> Any:
    # TimesFM.__new__ returns a version-specific subclass. Reconstruct through
    # the public class so a local path is classified the same way a user would.
    if type(model).__name__.startswith("_TimesFM"):
        from foundationforecast.models.timesfm import TimesFM

        return TimesFM
    return type(model)


def _clone_with_repo_id(model: Any, repo_id: str) -> Any:
    cls = _public_class(model)
    constructor = cls.__new__ if cls.__name__ == "TimesFM" else cls.__init__
    signature = inspect.signature(constructor)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name in {"self", "cls"} or param.kind in (
            param.VAR_KEYWORD,
            param.VAR_POSITIONAL,
        ):
            continue
        if name == "repo_id":
            kwargs[name] = repo_id
        elif hasattr(model, name):
            kwargs[name] = getattr(model, name)
    extra = getattr(model, "kwargs", None)
    if cls.__name__ == "TimesFM" and isinstance(extra, dict):
        kwargs.update(extra)
    return cls(**kwargs)


def _materialize_repo(snapshot: Path, local: Path) -> None:
    """Copy the repo layout without leaving Python files as cache-blob symlinks.

    Hugging Face snapshots symlink each file at ``blobs/<hash>``. Transformers
    resolves those links and then looks for relative imports next to the blob,
    so a dynamic module such as Tabby cannot load from the snapshot directory.
    """
    local.mkdir(parents=True, exist_ok=True)
    for entry in snapshot.iterdir():
        dest = local / entry.name
        if entry.is_dir():
            _materialize_repo(entry.resolve(), dest)
            continue
        resolved = entry.resolve()
        if dest.suffix == ".py":
            try:
                os.link(resolved, dest)
            except OSError:
                shutil.copy2(resolved, dest)
        else:
            dest.symlink_to(resolved)


def _link_local_repo(repo_id: str, directory: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id))
    local = directory / repo_id.rsplit("/", 1)[-1]
    _materialize_repo(snapshot, local)
    return local


def test_local_repo_keeps_python_files_out_of_the_cache_blobs(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    (blobs / "modeling").write_text("from .sibling import x\n", encoding="utf-8")
    (blobs / "sibling").write_text("x = 1\n", encoding="utf-8")
    (blobs / "weights").write_bytes(b"weights")
    (snapshot / "modeling.py").symlink_to(blobs / "modeling")
    (snapshot / "sibling.py").symlink_to(blobs / "sibling")
    (snapshot / "model.safetensors").symlink_to(blobs / "weights")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id: str(snapshot),
    )

    local = _link_local_repo("paris-noah/Tabby", tmp_path / "repos")

    assert not (local / "modeling.py").is_symlink()
    assert (local / "sibling.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (local / "model.safetensors").is_symlink()
    assert (local / "model.safetensors").resolve() == (blobs / "weights").resolve()


@pytest.mark.models
@pytest.mark.parametrize("model", models, ids=_local_repo_case_id)
def test_forecast_from_local_checkpoint(model, tmp_path):
    """Every registered checkpoint loads from a local directory, not only a Hub id."""
    checkpoint = getattr(model, "repo_id", None)
    if not isinstance(checkpoint, str):
        pytest.skip("model has no Hugging Face checkpoint")
    local_model = _clone_with_repo_id(
        model, str(_link_local_repo(cast(str, checkpoint), tmp_path))
    )
    df = generate_series(n_series=1, freq="D", min_length=64, max_length=64)
    fcst = local_model.forecast(df, h=2, freq="D")
    assert fcst.shape[0] == 2
    assert local_model.alias in fcst.columns
    assert np.isfinite(fcst[local_model.alias]).all()
