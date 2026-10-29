import numpy as np
import pytest

from foundationforecast.core.conformal import (
    ConformalIntervals,
    add_conformal_distribution_intervals,
    add_conformal_error_intervals,
    conformity_scores_abs_errors,
    effective_n_windows,
)


def test_conformal_intervals_validation():
    with pytest.raises(ValueError, match="at least two windows"):
        ConformalIntervals(n_windows=1)
    with pytest.raises(ValueError, match="method must be one of"):
        ConformalIntervals(method="invalid")


def test_effective_n_windows_caps_by_length():
    assert effective_n_windows(100, h=3, n_windows=10) == 10
    assert effective_n_windows(10, h=3, n_windows=10) == 3


def test_conformity_scores_abs_errors_constant_forecast():
    y = np.arange(20, dtype=float)
    h = 2
    n_windows = 2

    def forecast_values(train_end: int, _i_window: int) -> np.ndarray:
        return np.full(h, float(train_end))

    cs = conformity_scores_abs_errors(y, h, n_windows, forecast_values)
    assert cs.shape == (2, 2)
    assert np.all(cs >= 0)


def test_add_conformal_error_intervals_symmetric():
    mean = np.array([10.0, 20.0])
    cs = np.array([[1.0, 2.0], [3.0, 4.0]])
    bands = add_conformal_error_intervals(mean, cs, [90])
    assert bands["lo-90"][0] == pytest.approx(10.0 - np.quantile(cs[:, 0], 0.9))
    assert bands["hi-90"][1] == pytest.approx(20.0 + np.quantile(cs[:, 1], 0.9))


def test_add_conformal_distribution_intervals_shape():
    mean = np.array([5.0, 5.0])
    cs = np.array([[1.0, 1.0], [2.0, 2.0]])
    bands = add_conformal_distribution_intervals(mean, cs, [80, 90])
    assert set(bands) == {"lo-90", "lo-80", "hi-80", "hi-90"}
    assert len(bands["lo-80"]) == 2
