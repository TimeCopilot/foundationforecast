import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.exog import (
    XReg,
    infer_futr_exog_columns,
    normalize_exog_strategy,
    prepare_futr_exog_context,
    resolve_horizon_exog_df,
    validate_exog_strategy_for_timegpt,
    validate_futr_exog_inputs,
)
from foundationforecast.core.exog.xreg import (
    adjust_point_forecast_with_xreg,
    merge_xreg_into_forecast_df,
)
from foundationforecast.models.tafsut import Tafsut


def _panel(n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    exog = np.linspace(0, 1, n)
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * n,
            "ds": ds,
            "y": exog * 2 + np.random.default_rng(0).normal(scale=0.01, size=n),
            "x1": exog,
        }
    )
    h = 3
    futr_ds = pd.date_range(ds[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    X_df = pd.DataFrame(
        {
            "unique_id": ["A"] * h,
            "ds": futr_ds,
            "x1": [0.9, 0.95, 1.0],
        }
    )
    return df, X_df


def test_infer_futr_exog_columns():
    X_df = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2020-01-02")],
            "x1": [1.0],
            "x2": [2.0],
        }
    )
    assert infer_futr_exog_columns(X_df) == ["x1", "x2"]


def test_resolve_horizon_exog_df_alias():
    df = pd.DataFrame({"a": [1]})
    assert resolve_horizon_exog_df(df, None) is df
    with pytest.raises(ValueError, match="same DataFrame"):
        resolve_horizon_exog_df(df, pd.DataFrame({"a": [2]}))


def test_futr_exog_list_without_x_df_raises():
    model = Tafsut()
    df = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": [pd.Timestamp("2020-01-01")],
            "y": [1.0],
            "x1": [0.5],
        }
    )
    with pytest.raises(ValueError, match="without X_df"):
        prepare_futr_exog_context(
            model, df, 1, X_df=None, futr_df=None, futr_exog_list=["x1"]
        )


def test_validate_futr_exog_inputs():
    df, X_df = _panel()
    validate_futr_exog_inputs(df, 3, X_df, ["x1"])


def test_normalize_exog_strategy_rejects_typos():
    with pytest.raises(ValueError, match="Invalid exog_strategy"):
        normalize_exog_strategy("natvie")  # type: ignore[arg-type]


def test_validate_missing_series_in_x_df():
    df, X_df = _panel()
    X_df = X_df.copy()
    X_df["unique_id"] = "B"
    with pytest.raises(ValueError, match="not present in df"):
        validate_futr_exog_inputs(df, 3, X_df, ["x1"])


def test_validate_duplicate_horizon_keys():
    df, X_df = _panel()
    X_df = pd.concat([X_df, X_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_futr_exog_inputs(df, 3, X_df, ["x1"])


def test_xreg_adjusts_probabilistic_columns():
    df, X_df = _panel()
    point = np.array([1.0, 2.0, 3.0])
    fcst = pd.DataFrame(
        {
            "unique_id": X_df["unique_id"],
            "ds": X_df["ds"],
            "Tafsut": point,
            "Tafsut-lo-80": point - 0.5,
            "Tafsut-hi-80": point + 0.5,
        }
    )
    out = merge_xreg_into_forecast_df(
        fcst, "Tafsut", df, X_df, ["x1"], h=3, xreg=XReg(fm_first=True)
    )
    delta = out["Tafsut"].to_numpy() - point
    assert np.allclose(out["Tafsut-lo-80"].to_numpy(), fcst["Tafsut-lo-80"] + delta)
    assert np.allclose(out["Tafsut-hi-80"].to_numpy(), fcst["Tafsut-hi-80"] + delta)
    assert not np.allclose(delta, 0)


def test_validate_wrong_h_per_series():
    df, X_df = _panel()
    X_df = X_df.iloc[:2]
    with pytest.raises(ValueError, match="exactly h"):
        validate_futr_exog_inputs(df, 3, X_df, ["x1"])


def test_timegpt_rejects_xreg_strategy():
    with pytest.raises(ValueError, match="TimeGPT"):
        validate_exog_strategy_for_timegpt(XReg())


def test_xreg_adjustment_depends_on_exog():
    rng = np.random.default_rng(1)
    n = 30
    x_hist = rng.normal(size=(n, 1))
    x_hor = rng.normal(size=(5, 1))
    y = (x_hist @ np.array([[2.0]]) + rng.normal(scale=0.01, size=(n, 1))).ravel()
    base = np.linspace(0, 1, 5)
    out1 = adjust_point_forecast_with_xreg(
        y_hist=y,
        exog_hist=x_hist,
        exog_horizon=x_hor,
        baseline_horizon=base,
        xreg=XReg(fm_first=True),
    )
    out2 = adjust_point_forecast_with_xreg(
        y_hist=y,
        exog_hist=x_hist,
        exog_horizon=x_hor + 1.0,
        baseline_horizon=base,
        xreg=XReg(fm_first=True),
    )
    assert not np.allclose(out1, out2)


@pytest.mark.slow
def test_tafsut_xreg_forecast_changes_with_x_df():
    df, X_df = _panel()
    model = Tafsut()
    u = model.forecast(df=df, h=3, freq="D")
    v = model.forecast(df=df, h=3, freq="D", X_df=X_df)
    assert not np.allclose(u["Tafsut"].to_numpy(), v["Tafsut"].to_numpy())


def test_tafsut_native_strategy_raises_with_x_df():
    df, X_df = _panel()
    model = Tafsut()
    model.exog_strategy = "native"
    with pytest.raises(ValueError, match="native"):
        model.forecast(df=df, h=3, freq="D", X_df=X_df)


def test_cross_validation_reads_horizon_exog_from_df():
    from tests.helpers import (
        generate_panel_with_futr_exog,
        panel_with_futr_exog_horizon,
    )

    h = 3
    df_hist, X_df, futr_exog_list = generate_panel_with_futr_exog(
        1, freq="D", h=h, min_length=32, max_length=32
    )
    panel = panel_with_futr_exog_horizon(df_hist, X_df)
    model = Tafsut()
    cv = model.cross_validation(
        df=panel, h=h, freq="D", n_windows=1, futr_exog_list=futr_exog_list
    )
    assert len(cv) == h
    assert model.alias in cv.columns
