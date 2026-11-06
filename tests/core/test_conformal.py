import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.conformal import (
    ConformalIntervals,
    add_conformal_distribution_intervals,
    add_conformal_error_intervals,
    get_conformal_method,
    validate_conformal_samples,
)


def test_conformal_intervals_validation():
    with pytest.raises(ValueError, match="at least two windows"):
        ConformalIntervals(n_windows=1)
    with pytest.raises(ValueError, match="method must be one of"):
        ConformalIntervals(method="invalid")
    with pytest.raises(ValueError, match="step_size must be a positive integer"):
        ConformalIntervals(step_size=0)


def test_validate_conformal_samples():
    validate_conformal_samples(7, h=3, n_windows=2, step_size=3)
    with pytest.raises(ValueError, match="Minimum required samples"):
        validate_conformal_samples(6, h=3, n_windows=2, step_size=3)
    validate_conformal_samples(5, h=3, n_windows=2, step_size=1)


def _fcst_cs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fcst = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "M": [10.0, 20.0],
        }
    )
    cs = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]
            ),
            "cutoff": pd.to_datetime(["2019-12-31"] * 2 + ["2020-01-01"] * 2),
            "M": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return fcst, cs


def test_add_conformal_error_intervals_symmetric():
    fcst, cs = _fcst_cs()
    out = add_conformal_error_intervals(fcst, cs, ["M"], [90], 2, 2, 1, 2)
    q = np.quantile(cs["M"].to_numpy().reshape(2, 2), 0.9, axis=0)
    assert out["M-lo-90"].to_numpy() == pytest.approx([10.0, 20.0] - q)
    assert out["M-hi-90"].to_numpy() == pytest.approx([10.0, 20.0] + q)


def test_add_conformal_distribution_intervals_shape():
    fcst, cs = _fcst_cs()
    out = add_conformal_distribution_intervals(fcst, cs, ["M"], [80, 90], 2, 2, 1, 2)
    assert {"M-lo-90", "M-lo-80", "M-hi-80", "M-hi-90"} <= set(out.columns)
    assert len(out["M-lo-80"]) == 2


@pytest.mark.parametrize(
    "level, radius",
    [(20, 1.0), (40, 2.0), (60, 3.0), (80, 4.0), (90, np.inf)],
)
def test_finite_sample_uses_exact_order_statistic(level, radius):
    fcst, _ = _fcst_cs()
    # Deliberately unsorted, with different scales at each horizon step.
    cs = pd.DataFrame({"M": [3.0, 30.0, 1.0, 10.0, 4.0, 40.0, 2.0, 20.0]})
    out = get_conformal_method("conformal_error_finite_sample")(
        fcst, cs, ["M"], [level], 4, 2, 1, 2
    )
    expected = np.array([radius, 10 * radius])
    np.testing.assert_array_equal(out[f"M-lo-{level}"], [10.0, 20.0] - expected)
    np.testing.assert_array_equal(out[f"M-hi-{level}"], [10.0, 20.0] + expected)
    assert list(fcst.columns) == ["unique_id", "ds", "M"]


@pytest.mark.parametrize("n_windows", [18, 19, 20])
def test_finite_sample_95_percent_boundary(n_windows):
    fcst, _ = _fcst_cs()
    cs = pd.DataFrame({"M": np.repeat(np.arange(1, n_windows + 1), 2)})
    out = get_conformal_method("conformal_error_finite_sample")(
        fcst, cs, ["M"], [95], n_windows, 2, 1, 2
    )
    radius = np.inf if n_windows == 18 else n_windows
    np.testing.assert_array_equal(out["M-hi-95"], fcst["M"] + radius)
    np.testing.assert_array_equal(out["M-lo-95"], fcst["M"] - radius)


def test_finite_sample_keeps_series_horizons_and_models_separate():
    fcst = pd.DataFrame({"M": [10.0, 20.0, 30.0, 40.0]})
    fcst["N"] = fcst["M"] + 100
    # (window, series, horizon); the third horizon step is not requested.
    scores = np.array(
        [[[3, 30, 999], [300, 3000, 999]], [[1, 10, 999], [100, 1000, 999]]]
    )
    cs = pd.DataFrame({"M": scores.ravel(), "N": 2 * scores.ravel()})
    out = get_conformal_method("conformal_error_finite_sample")(
        fcst, cs, ["M", "N"], [50, 80], 2, 3, 2, 2
    )
    for model, scale in [("M", 1), ("N", 2)]:
        radii = scale * np.array([3, 30, 300, 3000])
        np.testing.assert_array_equal(out[f"{model}-hi-50"], fcst[model] + radii)
        np.testing.assert_array_equal(out[f"{model}-lo-50"], fcst[model] - radii)
        assert np.isposinf(out[f"{model}-hi-80"]).all()
        assert np.isneginf(out[f"{model}-lo-80"]).all()


@pytest.mark.parametrize("n_windows, level", [(2, 80), (4, 80), (9, 80), (19, 95)])
def test_finite_sample_exchangeable_rank_coverage(n_windows, level):
    # Every leave-one-out position is equally likely for exchangeable errors.
    errors = np.arange(1, n_windows + 2, dtype=float)
    covered = []
    method = get_conformal_method("conformal_error_finite_sample")
    for held_out in range(n_windows + 1):
        out = method(
            pd.DataFrame({"M": [0.0]}),
            pd.DataFrame({"M": np.delete(errors, held_out)}),
            ["M"],
            [level],
            n_windows,
            1,
            1,
            1,
        )
        covered.append(errors[held_out] <= out[f"M-hi-{level}"].iloc[0])
    assert np.mean(covered) >= level / 100
    assert sum(covered) == min(
        n_windows + 1, int(np.ceil((n_windows + 1) * level / 100))
    )


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_finite_sample_rejects_nonfinite_calibration_errors(bad_score):
    fcst, cs = _fcst_cs()
    cs.loc[0, "M"] = bad_score
    with pytest.raises(ValueError, match="finite calibration errors"):
        get_conformal_method("conformal_error_finite_sample")(
            fcst, cs, ["M"], [50], 2, 2, 1, 2
        )


@pytest.mark.parametrize("level", [-1, 101, np.nan])
def test_finite_sample_rejects_out_of_range_levels(level):
    fcst, cs = _fcst_cs()
    with pytest.raises(ValueError, match=r"range \[0, 1\]"):
        get_conformal_method("conformal_error_finite_sample")(
            fcst, cs, ["M"], [level], 2, 2, 1, 2
        )


def test_finite_sample_tied_zero_errors_and_integer_percent_float():
    fcst, cs = _fcst_cs()
    cs["M"] = 0.0
    out = get_conformal_method("conformal_error_finite_sample")(
        fcst, cs, ["M"], [50.0], 2, 2, 1, 2
    )
    np.testing.assert_array_equal(out["M-lo-50.0"], fcst["M"])
    np.testing.assert_array_equal(out["M-hi-50.0"], fcst["M"])
