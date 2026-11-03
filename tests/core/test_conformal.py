import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.conformal import (
    ConformalIntervals,
    add_conformal_distribution_intervals,
    add_conformal_error_intervals,
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
