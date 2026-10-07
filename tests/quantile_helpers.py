"""Test-only helpers for quantile forecast column names and monotonicity checks."""

from __future__ import annotations

from foundationforecast.core.quantiles import QUANTILE_GRID_TOLERANCE


def quantile_from_percent_suffix(suffix: str) -> float:
    """Parse the ``-q-{suffix}`` segment back to a quantile level."""
    if "." in suffix:
        whole, frac = suffix.split(".", 1)
        if len(frac) != 1:
            msg = f"Invalid quantile column suffix {suffix!r}"
            raise ValueError(msg)
        m = int(whole) * 10 + int(frac)
    else:
        m = int(suffix) * 10
    return m / 1000


def quantile_from_column_name(model_alias: str, column: str) -> float:
    """Extract the quantile level from a model quantile forecast column."""
    prefix = f"{model_alias}-q-"
    if not column.startswith(prefix):
        msg = f"Column {column!r} is not a quantile column for {model_alias!r}"
        raise ValueError(msg)
    return quantile_from_percent_suffix(column[len(prefix) :])


def quantile_pair_may_equal_under_edge_clamp(
    q_lower: float,
    q_upper: float,
    q_min: float,
    q_max: float,
) -> bool:
    """Whether edge clamping can make ``q_lower`` and ``q_upper`` forecasts equal."""
    tol = QUANTILE_GRID_TOLERANCE
    return q_lower < q_min - tol or q_upper > q_max + tol
