import pandas as pd
import pytest

from tests.helpers import (
    generate_panel_with_futr_exog,
    generate_series,
    panel_with_futr_exog_horizon,
)
from .conftest import models
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
    for c1, c2 in zip(ordered_q_cols[:-1], ordered_q_cols[1:], strict=False):
        if "chronos" in model.alias.lower() or "median" in model.alias.lower():
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif "timesfm" in model.alias.lower() or "flowstate" in model.alias.lower():
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.8
        elif "tabpfn" in model.alias.lower():
            continue
        elif "moe" in model.alias.lower():
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.5
        elif "patchtst" in model.alias.lower() or "granite" in model.alias.lower():
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.8
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
    # 20/40/60/80 hit native decile knots; 50 → 0.25/0.75; 95 → 0.025/0.975 (interpolated)
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
