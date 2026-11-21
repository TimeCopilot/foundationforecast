import copy
import math

import numpy as np
import pandas as pd
import pytest

from foundationforecast import ACI, ConformalIntervals


def _errors(values):
    return np.asarray(values, dtype=float).reshape(-1, 1, 1)


def _oracle(scores, alpha, *, lag, step, gap, gamma, ncal, rolling):
    """Scalar calendar-time oracle, independent of the replay implementation."""
    origins = [k * step for k in range(len(scores))]
    final_origin = origins[-1] + gap
    state = alpha
    observed = []
    issued: dict[int, float] = {}
    result = math.inf
    for now in range(final_origin + 1):
        for k, origin in enumerate(origins):
            if origin + lag == now:
                observed.append(scores[k])
                if k in issued:
                    state += gamma * (alpha - float(scores[k] > issued[k]))
        decisions = [k for k in range(ncal, len(scores)) if origins[k] == now]
        if decisions or now == final_origin:
            if state >= 1:
                result = -math.inf
            elif state <= 0 or not observed:
                result = math.inf
            else:
                history = observed[-ncal:] if rolling else observed
                rank = math.ceil(len(history) * (1 - state))
                result = sorted(history)[max(0, rank - 1)]
            for k in decisions:
                issued[k] = result
    return result


def test_hand_computed_aci_update_and_empirical_rank():
    # Warm up on 1,2, then level 50 has radius 1 (inverse empirical CDF).
    # 10 misses -> alpha=.25, radius=10; 3 is covered -> alpha=.5, radius=2.
    method = ACI(gamma=0.5, ncal=2)
    result = method.thresholds(_errors([1, 2, 10, 3]), [50], 1, 1)
    np.testing.assert_array_equal(result, [[[2.0]]])


def test_delayed_feedback_scores_the_interval_issued_at_its_origin():
    errors = np.repeat(_errors([1, 2, 10, 3]), 2, axis=2)
    result = ACI(gamma=0.5, ncal=2).thresholds(errors, [50], 1, 2)
    # At lag 2, both 10 and 3 miss the radius 1 issued before either was known.
    # Alpha falls to zero, so the final radius is infinite. Lag 1 remains 2.
    np.testing.assert_array_equal(result, [[[2.0, np.inf]]])


@pytest.mark.parametrize("step,gap", [(1, 3), (2, 3), (4, 3), (2, 1)])
@pytest.mark.parametrize("rolling", [False, True])
def test_replay_matches_independent_calendar_time_oracle(step, gap, rolling):
    errors = np.random.default_rng(7).normal(size=(18, 2, 3))
    levels = [50, 80, 95]
    method = ACI(gamma=0.07, ncal=3, rolling=rolling)
    actual = method.thresholds(errors, levels, step, gap)
    for i, level in enumerate(levels):
        for series in range(2):
            for j in range(3):
                expected = _oracle(
                    np.abs(errors[:, series, j]).tolist(),
                    1 - level / 100,
                    lag=j + 1,
                    step=step,
                    gap=gap,
                    gamma=0.07,
                    ncal=3,
                    rolling=rolling,
                )
                assert actual[i, series, j] == expected


def test_unrevealed_errors_cannot_change_the_forecast_threshold():
    errors = np.random.default_rng(11).normal(size=(12, 1, 4))
    method = ACI(ncal=2)
    before = method.thresholds(errors, [80], 1, 1)
    # At origin 12, horizon-4 errors from origins 9,10,11 are still unavailable.
    perturbed = errors.copy()
    perturbed[9:, :, 3] = 1e9
    after = method.thresholds(perturbed, [80], 1, 1)
    np.testing.assert_array_equal(before, after)


def test_series_horizons_and_levels_are_independent():
    errors = np.random.default_rng(2).normal(size=(20, 3, 2))
    method = ACI(ncal=5, gamma=0.02)
    joint = method.thresholds(errors, [50, 80, 95], 1, 2)
    for i, level in enumerate([50, 80, 95]):
        for series in range(3):
            for j in range(2):
                # Same horizon is retained so its observation delay is unchanged.
                single = method.thresholds(
                    errors[:, series : series + 1], [level], 1, 2
                )
                assert joint[i, series, j] == single[0, 0, j]


def test_replay_has_no_state_between_calls_and_does_not_mutate_errors():
    errors = np.random.default_rng(3).normal(size=(20, 2, 2))
    original = errors.copy()
    method = ACI(gamma=0.02, ncal=5)
    configuration = copy.deepcopy(vars(method))
    first = method.thresholds(errors, [80, 95], 2, 2)
    method.thresholds(100 * errors, [50], 1, 2)
    second = method.thresholds(errors, [80, 95], 2, 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(errors, original)
    assert vars(method) == configuration


def _call(method, errors, levels, models=("M",)):
    n_windows, n_series, h = errors.shape
    forecasts = pd.DataFrame({model: np.arange(n_series * h) for model in models})
    calibration = pd.DataFrame({model: errors.ravel() for model in models})
    return method(
        forecasts, calibration, list(models), levels, n_windows, h, n_series, h, h
    )


@pytest.mark.filterwarnings("error:ACI replay.*:UserWarning")
def test_finite_bounds_are_symmetric_nested_and_keep_point_forecasts():
    errors = np.random.default_rng(4).normal(size=(20, 2, 3))
    result = _call(ACI(gamma=0.005), errors, [50, 80, 95])
    assert list(result) == [
        "M",
        "M-lo-95",
        "M-lo-80",
        "M-lo-50",
        "M-hi-50",
        "M-hi-80",
        "M-hi-95",
    ]
    np.testing.assert_array_equal(result["M"], np.arange(6))
    assert np.isfinite(result).all().all()
    for level in [50, 80, 95]:
        np.testing.assert_allclose(
            result["M"] - result[f"M-lo-{level}"], result[f"M-hi-{level}"] - result["M"]
        )
    assert result["M-lo-95"].le(result["M-lo-80"]).all()
    assert result["M-lo-80"].le(result["M-lo-50"]).all()
    assert result["M-hi-95"].ge(result["M-hi-80"]).all()
    assert result["M-hi-80"].ge(result["M-hi-50"]).all()


def test_unbounded_intervals_warn_once_across_models():
    errors = _errors(np.arange(1, 13))
    with pytest.warns(UserWarning, match="ACI.*unbounded.*95") as caught:
        result = _call(ACI(gamma=0.5, ncal=2), errors, [95], models=("M", "N"))
    assert len(caught) == 1
    assert "n_windows=12" in str(caught[0].message)
    for model in ["M", "N"]:
        assert np.isneginf(result[f"{model}-lo-95"]).all()
        assert np.isposinf(result[f"{model}-hi-95"]).all()


def test_alpha_is_not_clipped_and_empty_intervals_are_explicit():
    # With all-zero errors, alpha=.5 -> 1 (cover) -> .5 (empty/miss) -> 1.
    errors = _errors([0, 0, 0, 0])
    raw = ACI(gamma=1, ncal=1).thresholds(errors, [50], 1, 1)
    assert np.isneginf(raw).all()
    with pytest.warns(UserWarning, match="ACI.*empty.*50"):
        result = _call(ACI(gamma=1, ncal=1), errors, [50])
    assert np.isposinf(result["M-lo-50"]).all()
    assert np.isneginf(result["M-hi-50"]).all()


@pytest.mark.parametrize(
    "level,radius,warning", [(0, -np.inf, "empty"), (100, np.inf, "unbounded")]
)
def test_boundary_levels_have_defined_set_semantics(level, radius, warning):
    method = ACI(ncal=1)
    assert method.thresholds(_errors([1, 2]), [level], 1, 1)[0, 0, 0] == radius
    with pytest.warns(UserWarning, match=warning):
        _call(method, _errors([1, 2]), [level])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0},
        {"gamma": -1},
        {"gamma": 1.1},
        {"gamma": np.nan},
        {"gamma": np.inf},
        {"gamma": True},
        {"ncal": 0},
        {"ncal": 1.5},
        {"ncal": True},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ACI(**kwargs)


def test_conformal_intervals_validate_the_warmup_period():
    ConformalIntervals(n_windows=2, method=ACI())
    with pytest.raises(ValueError, match="ncal.*smaller"):
        ConformalIntervals(n_windows=4, method=ACI(ncal=4))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_errors_are_rejected(bad):
    errors = _errors([1, 2, 3, 4])
    errors[1, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        ACI().thresholds(errors, [80], 1, 1)


@pytest.mark.parametrize("level", [[], [-1], [101], [np.nan]])
def test_invalid_levels_are_rejected(level):
    with pytest.raises(ValueError, match="level"):
        ACI().thresholds(_errors([1, 2]), level, 1, 1)


@pytest.mark.parametrize("step,gap", [(0, 1), (1, 0), (1.5, 1)])
def test_invalid_calendar_spacing_is_rejected(step, gap):
    with pytest.raises(ValueError, match="positive integer"):
        ACI().thresholds(_errors([1, 2]), [80], step, gap)


@pytest.mark.parametrize("shape", [(2, 2), (0, 1, 1), (2, 0, 1), (2, 1, 0)])
def test_invalid_calibration_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="shape"):
        ACI().thresholds(np.zeros(shape), [80], 1, 1)


def test_single_window_cannot_seed_and_replay():
    with pytest.raises(ValueError, match="at least 2"):
        ACI().thresholds(_errors([1]), [80], 1, 1)


def test_out_of_sample_replay_responds_to_a_variance_shift():
    rng = np.random.default_rng(19)
    errors = rng.normal(size=300) * np.where(np.arange(300) < 150, 1.0, 4.0)
    method = ACI(gamma=0.02, ncal=30)
    misses = []
    for end in range(150, len(errors)):
        radius = method.thresholds(_errors(errors[:end]), [90], 1, 1)[0, 0, 0]
        misses.append(abs(errors[end]) > radius)
    static_radius = np.quantile(np.abs(errors[:150]), 0.9)
    static_misses = np.mean(np.abs(errors[150:]) > static_radius)
    assert static_misses > 0.3
    assert abs(np.mean(misses) - 0.1) < 0.06
