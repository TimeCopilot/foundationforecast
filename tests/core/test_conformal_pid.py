import math

import numpy as np
import pandas as pd
import pytest

from foundationforecast import PID, ConformalIntervals
from foundationforecast.core.conformal_pid import default_csat, track_quantiles


def _track(scores, alpha, *, step=1, gap=1, **overrides):
    """Run the tracker on a single-series, single-horizon score vector."""
    scores = np.asarray(scores, dtype=float)
    kwargs = dict(
        step_size=step,
        final_gap=gap,
        lr=1.0,
        ki=np.zeros(1),
        csat=1.0,
        ncal=2,
        rolling=False,
        integrate=False,
        warm_start=False,
    )
    kwargs.update(overrides)
    out = track_quantiles(
        scores.reshape(len(scores), 1, -1), np.atleast_1d(alpha), **kwargs
    )
    return out


def _oracle_tail(
    scores,
    alpha,
    lag,
    *,
    step,
    gap,
    lr,
    ki=0.0,
    csat=1.0,
    integrate=False,
    rolling=False,
    ncal=2,
):
    """Scalar, calendar-time reference: an error counts once ``origin + lag`` passes."""
    n = len(scores)
    origins = [k * step for k in range(n)] + [(n - 1) * step + gap]
    p, integral, misses, n_obs = 0.0, 0.0, 0.0, 0
    issued: list[float] = []
    done: set[int] = set()
    for now in origins:
        for k in range(n):
            if k in done or origins[k] + lag > now:
                continue
            done.add(k)
            miss = float(scores[k] > issued[k])
            hist = scores[max(0, k - ncal + 1) : k + 1] if rolling else scores[: k + 1]
            eta = lr if len(hist) <= 1 else lr * (max(hist) - min(hist))
            p += eta * (miss - alpha)
            misses += miss
            n_obs += 1
            if integrate and n_obs > 1 and ki > 0:
                x = (misses - n_obs * alpha) * math.log(n_obs) / (csat * n_obs)
                if x >= math.pi / 2:
                    integral = math.inf
                elif x <= -math.pi / 2:
                    integral = -math.inf
                else:
                    integral = ki * math.tan(x)
        issued.append(p + integral)
    return issued[-1]


def test_quantile_tracking_golden_value():
    # alpha=0.2, lr=1, scores 1,3,2 (hand computed):
    #   window 0 misses q=0   -> eta=1 (single score)    p = 0   + 1 * 0.8 = 0.8
    #   window 1 misses q=0.8 -> eta=3-1=2               p = 0.8 + 2 * 0.8 = 2.4
    #   window 2 covered q=2.4 -> eta=3-1=2              p = 2.4 + 2 * -0.2 = 2.0
    out = _track([1.0, 3.0, 2.0], 0.2)
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == pytest.approx(2.0)


def test_quantile_tracking_delays_errors_by_horizon():
    # Same scores but each error is revealed two steps after its origin, so
    # windows 1 and 2 are judged against the stale threshold of window 1 / 2:
    #   window 0 -> 0.8 (issued 0.8 at origin 2), window 1 missed q=0 -> 2.4,
    #   window 2 missed q=0.8 -> 2.4 + 2 * 0.8 = 4.0
    scores = np.array([1.0, 3.0, 2.0]).reshape(3, 1, 1)
    scores = np.concatenate([scores, scores], axis=2)  # horizons 1 and 2
    out = track_quantiles(
        scores,
        np.array([0.2]),
        step_size=1,
        final_gap=2,
        lr=1.0,
        ki=np.zeros(1),
        csat=1.0,
        ncal=2,
        rolling=False,
        integrate=False,
        warm_start=False,
    )
    assert out[0, 0, 0] == pytest.approx(2.0)
    assert out[0, 0, 1] == pytest.approx(4.0)


def test_error_integration_golden_value():
    scores = [1.0, 3.0, 2.0]
    base = _track(scores, 0.2)[0, 0, 0]
    out = _track(scores, 0.2, integrate=True, ki=np.array([2.0]), csat=1.0)[0, 0, 0]
    # two misses over three observed errors -> sum(err - alpha) = 2 - 0.6
    expected_integral = 2.0 * math.tan(1.4 * math.log(3) / (1.0 * 3))
    assert out - base == pytest.approx(expected_integral)


def test_integrator_saturates_to_infinity_when_far_from_target():
    out = _track(
        np.arange(1.0, 40.0),
        0.5,
        integrate=True,
        ki=np.array([1.0]),
        csat=0.01,
    )
    assert np.isinf(out).all()


def test_zero_ki_disables_integrator():
    scores = np.random.default_rng(0).normal(size=30) ** 2
    no_int = _track(scores, 0.1)
    zero_ki = _track(scores, 0.1, integrate=True, ki=np.zeros(1))
    np.testing.assert_allclose(no_int, zero_ki)


def test_warm_start_with_tiny_lr_returns_empirical_quantile():
    scores = np.random.default_rng(1).uniform(size=40)
    out = _track(scores, 0.2, lr=1e-12, ncal=15, warm_start=True)
    assert out[0, 0, 0] == pytest.approx(np.quantile(scores[:15], 0.8), abs=1e-6)


@pytest.mark.parametrize("step,gap,lag", [(1, 1, 1), (1, 3, 3), (2, 3, 3), (3, 3, 2)])
@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("integrate", [False, True])
def test_vectorized_tracker_matches_calendar_time_oracle(
    step, gap, lag, rolling, integrate
):
    rng = np.random.default_rng(7)
    scores = rng.normal(size=60) * np.linspace(1, 3, 60)
    alpha = 0.1
    ki = float(np.abs(scores).max())
    csat = default_csat(60, 0.01)
    horizon_scores = np.stack([scores] * gap, axis=1).reshape(60, 1, gap)
    out = track_quantiles(
        horizon_scores,
        np.array([alpha]),
        step_size=step,
        final_gap=gap,
        lr=0.1,
        ki=np.array([ki]),
        csat=csat,
        ncal=10,
        rolling=rolling,
        integrate=integrate,
        warm_start=False,
    )
    expected = _oracle_tail(
        scores,
        alpha,
        lag,
        step=step,
        gap=gap,
        lr=0.1,
        ki=ki,
        csat=csat,
        integrate=integrate,
        rolling=rolling,
        ncal=10,
    )
    assert out[0, 0, lag - 1] == pytest.approx(expected)


def test_default_csat_matches_paper_formula():
    assert default_csat(1000, 0.01) == pytest.approx(
        2 / math.pi * (1 - 1 / math.log(1000))
    )
    with pytest.raises(ValueError, match="non-positive Csat"):
        default_csat(2, 0.01)


def _shifted_errors(n=500, seed=3):
    rng = np.random.default_rng(seed)
    scale = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    return rng.normal(size=n) * scale


def test_online_coverage_recovers_after_distribution_shift():
    # h=1, step=1: the prefix run is exactly what was known at each origin, so
    # miss indicators on the next error are genuine out-of-sample coverage.
    errors = _shifted_errors()
    method = PID(symmetric=True, init="quantile", ncal=50, lr=0.1)
    start = 250
    misses = []
    for n in range(start, len(errors) - 1):
        _, upper = method.thresholds(
            errors[:n].reshape(n, 1, 1), [90], step_size=1, final_gap=1
        )
        misses.append(abs(errors[n]) > upper[0, 0, 0])
    pid_miss = np.mean(misses)
    static = np.quantile(np.abs(errors[:start]), 0.9)
    static_miss = np.mean(np.abs(errors[start:-1]) > static)
    assert abs(pid_miss - 0.1) < 0.04
    assert static_miss > 0.3


def test_prefix_run_does_not_depend_on_future_errors():
    errors = _shifted_errors(120)
    method = PID(symmetric=False, ncal=20)
    n = 80
    first = method.thresholds(errors[:n].reshape(n, 1, 1), [80], 1, 1)
    perturbed = errors.copy()
    perturbed[n:] = 1e6
    second = method.thresholds(perturbed[:n].reshape(n, 1, 1), [80], 1, 1)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_asymmetric_tails_follow_error_sign():
    rng = np.random.default_rng(5)
    errors = np.abs(rng.normal(size=(60, 1, 1))) + 0.5  # always overforecast
    lower, upper = PID(ncal=20).thresholds(errors, [80], 1, 1)
    assert upper[0, 0, 0] > 0
    assert lower[0, 0, 0] < upper[0, 0, 0]


def test_symmetric_pid_is_symmetric():
    errors = np.random.default_rng(6).normal(size=(40, 3, 2))
    lower, upper = PID(symmetric=True, ncal=10).thresholds(errors, [80, 95], 2, 2)
    np.testing.assert_array_equal(lower, upper)
    assert lower.shape == (2, 3, 2)


def test_series_are_independent():
    rng = np.random.default_rng(8)
    errors = rng.normal(size=(30, 3, 2))
    method = PID(ncal=10)
    joint = method.thresholds(errors, [80], 2, 2)
    for i in range(3):
        single = method.thresholds(errors[:, i : i + 1], [80], 2, 2)
        np.testing.assert_allclose(joint[0][:, i : i + 1], single[0])
        np.testing.assert_allclose(joint[1][:, i : i + 1], single[1])


def test_pid_call_adds_interval_columns_around_mean():
    n_windows, n_series, h = 12, 2, 3
    rng = np.random.default_rng(9)
    errors = rng.normal(size=(n_windows, n_series, h))
    cs_df = pd.DataFrame({"M": errors.ravel()})
    mean = np.arange(n_series * h, dtype=float)
    fcst_df = pd.DataFrame({"M": mean})
    out = PID(ncal=4)(
        fcst_df,
        cs_df,
        model_names=["M"],
        level=[80, 95],
        cs_n_windows=n_windows,
        cs_h=h,
        n_series=n_series,
        horizon=h,
        step_size=h,
    )
    assert list(out.columns) == ["M", "M-lo-95", "M-lo-80", "M-hi-80", "M-hi-95"]
    assert (out["M-lo-80"] <= out["M-hi-80"]).all()
    assert (out["M-lo-95"] <= out["M-lo-80"]).all()
    assert (out["M-hi-95"] >= out["M-hi-80"]).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lr": 0}, "lr"),
        ({"KI": -1}, "KI"),
        ({"Csat": 0}, "Csat"),
        ({"Tg": 1}, "Tg"),
        ({"delta": 1}, "delta"),
        ({"ncal": 0}, "ncal"),
        ({"init": "random"}, "init"),
    ],
)
def test_pid_validates_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PID(**kwargs)


def test_conformal_intervals_rejects_default_pid_with_too_few_windows():
    # Tg defaults to n_windows, and two windows give a non-positive Csat.
    with pytest.raises(ValueError, match="non-positive Csat.*n_windows=2"):
        ConformalIntervals(n_windows=2, method=PID())
    ConformalIntervals(n_windows=3, method=PID())
    ConformalIntervals(n_windows=2, method=PID(integrate=False))
    ConformalIntervals(n_windows=2, method=PID(Csat=0.1))
    ConformalIntervals(n_windows=2, method=PID(Tg=100))


def test_pid_without_integrator_runs_with_two_windows():
    errors = np.array([1.0, -2.0]).reshape(2, 1, 1)
    lower, upper = PID(integrate=False).thresholds(errors, [80], 1, 1)
    assert np.isfinite(lower).all() and np.isfinite(upper).all()


def test_conformal_intervals_accepts_pid_and_validates_ncal():
    intervals = ConformalIntervals(n_windows=10, method=PID(lr=0.2))
    assert intervals.method == PID(lr=0.2)
    with pytest.raises(ValueError, match="ncal=10 must be smaller"):
        ConformalIntervals(n_windows=10, method=PID(ncal=10))
    ConformalIntervals(n_windows=10, method=PID(ncal=10, init="zero"))
    with pytest.raises(ValueError, match="method must be one of"):
        ConformalIntervals(n_windows=4, method="pid")


def _call(method, errors, level):
    n_windows, n_series, h = errors.shape
    return method(
        pd.DataFrame({"M": np.zeros(n_series * h)}),
        pd.DataFrame({"M": errors.ravel()}),
        model_names=["M"],
        level=level,
        cs_n_windows=n_windows,
        cs_h=h,
        n_series=n_series,
        horizon=h,
        step_size=h,
    )


def test_bounds_always_contain_point_forecast():
    errors = np.abs(np.random.default_rng(10).normal(size=(30, 1, 2))) + 1.0
    out = _call(PID(ncal=10, integrate=False, lr=0.5), errors, [50])
    assert (out["M-lo-50"] <= 0).all() and (out["M-hi-50"] >= 0).all()


def test_saturated_integrator_warns_and_yields_infinite_bound():
    # every calibration error exceeds the warm-start threshold
    errors = np.linspace(1, 20, 12).reshape(12, 1, 1)
    with pytest.warns(UserWarning, match="integration saturated"):
        out = _call(PID(ncal=2, symmetric=True, Csat=0.05), errors, [90])
    assert np.isinf(out["M-hi-90"]).all()
