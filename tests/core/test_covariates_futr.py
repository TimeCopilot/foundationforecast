import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.exog import (
    infer_futr_exog_columns,
    normalize_exog_strategy,
    prepare_futr_exog_context,
    resolve_futr_exog_list,
    resolve_horizon_exog_df,
    validate_futr_exog_inputs,
)
from foundationforecast.models.chronos import Chronos
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


def test_futr_exog_list_rejects_metadata_columns():
    df, X_df = _panel()
    with pytest.raises(ValueError, match="metadata columns"):
        resolve_futr_exog_list(X_df, ["y"])


def test_futr_exog_list_rejects_empty():
    df, X_df = _panel()
    with pytest.raises(ValueError, match="at least one"):
        resolve_futr_exog_list(X_df, [])


def test_normalize_exog_strategy_rejects_typos():
    with pytest.raises(ValueError, match="Invalid exog_strategy"):
        normalize_exog_strategy("natvie")  # type: ignore[arg-type]


def test_normalize_exog_strategy_rejects_invalid_types():
    with pytest.raises(ValueError, match="Invalid exog_strategy type"):
        normalize_exog_strategy(object())  # type: ignore[arg-type]


def test_normalize_exog_strategy_false():
    assert normalize_exog_strategy(False) is False  # type: ignore[arg-type]


def test_normalize_exog_strategy_native_alias():
    assert normalize_exog_strategy("native") == "auto"  # type: ignore[arg-type]


def test_exog_strategy_false_ignores_x_df():
    df, X_df = _panel()
    model = Tafsut(exog_strategy=False)  # type: ignore[arg-type]
    u = model.forecast(df=df, h=3, freq="D")
    v = model.forecast(df=df, h=3, freq="D", X_df=X_df)
    pd.testing.assert_frame_equal(u, v)


def test_prepare_futr_exog_context_sorts_reversed_x_df():
    df, X_df = _panel()
    X_df_rev = X_df.iloc[::-1].reset_index(drop=True)
    model = Chronos(repo_id="amazon/chronos-2", alias="Chronos-2")
    ctx = prepare_futr_exog_context(
        model, df, 3, X_df=X_df_rev, futr_df=None, futr_exog_list=None
    )
    assert ctx is not None
    assert ctx.horizon_df["ds"].is_monotonic_increasing


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


def test_validate_wrong_h_per_series():
    df, X_df = _panel()
    X_df = X_df.iloc[:2]
    with pytest.raises(ValueError, match="exactly h"):
        validate_futr_exog_inputs(df, 3, X_df, ["x1"])


def test_non_native_auto_raises_with_x_df():
    df, X_df = _panel()
    model = Tafsut()
    with pytest.raises(ValueError, match="does not support known-future"):
        model.forecast(df=df, h=3, freq="D", X_df=X_df)


@pytest.mark.slow
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
    model = Chronos(repo_id="amazon/chronos-2", alias="Chronos-2")
    cv = model.cross_validation(
        df=panel, h=h, freq="D", n_windows=1, futr_exog_list=futr_exog_list
    )
    assert len(cv) == h
    assert model.alias in cv.columns
