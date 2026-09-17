import numpy as np
import pytest

from foundationforecast.core.quantiles import (
    interpolate_quantiles,
    resolve_quantile_values,
    validate_levels,
)


def test_validate_levels_rejects_zero():
    with pytest.raises(ValueError, match="level=0"):
        validate_levels([0, 80])
    with pytest.raises(ValueError, match="level=0"):
        validate_levels([80, 0])


def test_validate_levels_passes():
    assert validate_levels([80, 95]) == [80, 95]
    assert validate_levels(None) is None


def test_interpolate_quantiles_exact_knots():
    knot_qs = [0.1, 0.5, 0.9]
    knot_values = np.array([1.0, 2.0, 3.0])
    result = interpolate_quantiles(knot_qs, knot_values, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


def test_interpolate_quantiles_midpoint():
    knot_qs = [0.1, 0.9]
    knot_values = np.array([0.0, 10.0])
    result = interpolate_quantiles(knot_qs, knot_values, [0.5])
    np.testing.assert_allclose(result, [5.0])


def test_interpolate_quantiles_edge_clamping():
    knot_qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    knot_values = np.arange(9.0)
    result = interpolate_quantiles(knot_qs, knot_values, [0.01, 0.99])
    np.testing.assert_allclose(result, [0.0, 8.0])


def test_interpolate_quantiles_multidimensional():
    knot_qs = [0.1, 0.5, 0.9]
    knot_values = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    result = interpolate_quantiles(knot_qs, knot_values, [0.5], axis=-1)
    assert result.shape == (2, 2, 1)
    np.testing.assert_allclose(result[..., 0], [[2.0, 5.0], [8.0, 11.0]])


def test_interpolate_quantiles_single_knot():
    knot_values = np.array([[[4.0], [5.0]], [[6.0], [7.0]]])
    result = interpolate_quantiles([0.5], knot_values, [0.1, 0.9])
    np.testing.assert_allclose(result, np.repeat(knot_values, 2, axis=-1))


def test_interpolate_quantiles_single_knot_non_default_axis():
    knot_values = np.array([[4.0, 5.0]])
    result = interpolate_quantiles([0.5], knot_values, [0.1, 0.9], axis=0)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[4.0, 5.0], [4.0, 5.0]])


def test_assign_quantile_forecasts_rejects_duplicate_columns():
    import pandas as pd

    from foundationforecast.core.forecaster import Forecaster

    fcst_df = pd.DataFrame({"unique_id": ["a"], "ds": [1], "m": [1.0]})
    with pytest.raises(ValueError, match="duplicate output column names"):
        Forecaster._assign_quantile_forecasts(
            fcst_df,
            "m",
            [0.151, 0.159],
            __import__("numpy").array([[1.0, 2.0]]),
        )


def test_resolve_quantile_values_subset():
    knot_qs = [0.1, 0.5, 0.9]
    knot_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = resolve_quantile_values(knot_qs, knot_values, [0.1, 0.9])
    np.testing.assert_allclose(result, [[1.0, 3.0], [4.0, 6.0]])


def test_resolve_quantile_values_interpolates():
    knot_qs = [0.1, 0.9]
    knot_values = np.array([0.0, 10.0])
    result = resolve_quantile_values(knot_qs, knot_values, [0.5])
    np.testing.assert_allclose(result, [5.0])
