import pytest

from tests.quantile_helpers import (
    quantile_from_column_name,
    quantile_from_percent_suffix,
    quantile_pair_may_equal_under_edge_clamp,
)
from foundationforecast.core.quantiles import quantile_column_name


@pytest.mark.parametrize(
    ("q", "suffix"),
    [
        (0.1, "10"),
        (0.025, "2.5"),
        (0.57, "57"),
        (0.975, "97.5"),
    ],
)
def test_quantile_column_name_roundtrip(q, suffix):
    assert quantile_from_percent_suffix(suffix) == q
    col = quantile_column_name("TiRex", q)
    assert quantile_from_column_name("TiRex", col) == q


def test_quantile_pair_may_equal_under_edge_clamp():
    q_min, q_max = 0.1, 0.9
    assert quantile_pair_may_equal_under_edge_clamp(0.025, 0.1, q_min, q_max)
    assert not quantile_pair_may_equal_under_edge_clamp(0.1, 0.2, q_min, q_max)
    assert quantile_pair_may_equal_under_edge_clamp(0.85, 0.975, q_min, q_max)
