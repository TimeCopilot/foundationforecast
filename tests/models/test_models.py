import pandas as pd
import pytest
from utilsforecast.data import generate_series as _generate_series

from .conftest import models


def generate_series(n_series, freq, **kwargs):
    df = _generate_series(n_series, freq, **kwargs)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


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
    qs = [round(i * 0.1, 1) for i in range(1, 10)]
    df = generate_series(n_series=3, freq="D")
    fcst_df = model.forecast(
        df=df,
        h=2,
        freq="D",
        quantiles=qs,
    )
    exp_qs_cols = [f"{model.alias}-q-{int(100 * q)}" for q in qs]
    assert len(exp_qs_cols) == len(fcst_df.columns) - 3
    assert all(col in fcst_df.columns for col in exp_qs_cols)
    assert not any(("-lo-" in col or "-hi-" in col) for col in fcst_df.columns)
    # test monotonicity of quantiles
    for c1, c2 in zip(exp_qs_cols[:-1], exp_qs_cols[1:], strict=False):
        if "chronos" in model.alias.lower() or "median" in model.alias.lower():
            # sometimes it gives this condition
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif "timesfm" in model.alias.lower():
            # TimesFM is a bit more lenient with the monotonicity condition
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.8
        elif "tabpfn" in model.alias.lower():
            # we are testing the mock mode, so we don't care about monotonicity
            continue
        elif "moe" in model.alias.lower():
            # MoE is a bit more lenient with the monotonicity condition
            assert fcst_df[c1].le(fcst_df[c2]).mean() >= 0.5
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()


@pytest.mark.parametrize("model", models)
def test_using_level(model):
    level = [0, 20, 40, 60, 80]  # corresponds to qs [0.1, 0.2, ..., 0.9]
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
    # test monotonicity of levels
    exp_lv_cols = exp_lv_cols[2:]  # remove level 0
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        if "chronos" in model.alias.lower() or "median" in model.alias.lower():
            # sometimes it gives this condition
            assert fcst_df[c1].le(fcst_df[c2]).all()
        elif "tabpfn" in model.alias.lower():
            # we are testing the mock mode, so we don't care about monotonicity
            continue
        else:
            assert fcst_df[c1].lt(fcst_df[c2]).all()
