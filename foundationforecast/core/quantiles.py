"""Quantile interpolation utilities for fixed-knot forecast models.

Fixed-knot models predict a discrete set of native quantile levels (knots).
User-requested ``level`` or ``quantiles`` values that do not match those knots
are obtained by piecewise linear interpolation along the quantile axis.

Edge clamping
-------------
When a requested quantile falls below the lowest native knot (or above the
highest), the forecast at the nearest edge knot is returned. This matches
``numpy.interp`` semantics: values are not extrapolated beyond the model's
native quantile range. For example, requesting ``q=0.01`` on a model with
knots ``[0.1, ..., 0.9]`` returns the same values as ``q=0.1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import interp1d

_LEVEL_ZERO_ERROR = (
    "`level=0` is not supported. Point forecasts are always returned in the "
    "model column. Pass standard confidence levels (e.g. `[80, 95]`) or use "
    "`quantiles=` directly."
)

FIXED_KNOT_QUANTILES_NOTE = (
    "This model predicts a fixed set of native quantile knots internally, then "
    "linearly interpolates (with edge clamping) to any requested ``level`` or "
    "``quantiles``. See ``foundationforecast.core.quantiles`` for details."
)

# Documented native quantile ranges for backends that evaluate quantiles directly.
PATCHTST_FM_QUANTILE_RANGE = (0.01, 0.99)
T0_ALPHA_QUANTILE_RANGE = (0.1, 0.9)
T0_BETA_QUANTILE_RANGE = (0.01, 0.99)
DEFAULT_NATIVE_QUANTILE_RANGE = (0.01, 0.99)


def validate_levels(level: Sequence[int | float] | None) -> list[int | float] | None:
    """Validate levels, rejecting the legacy ``level=0`` sentinel."""
    if level is None:
        return None
    levels = list(level)
    if any(lv == 0 for lv in levels):
        raise ValueError(_LEVEL_ZERO_ERROR)
    return levels


def _match_quantile_level(levels: Sequence[float], q: float) -> int:
    arr = np.asarray(levels, dtype=np.float64)
    matches = np.where(np.isclose(arr, q, rtol=0.0, atol=1e-9))[0]
    if matches.size == 0:
        raise ValueError(
            f"Quantile level {q} missing from backend output levels {list(levels)}"
        )
    return int(matches[0])


def clip_quantiles_to_range(
    quantiles: Sequence[float],
    q_min: float,
    q_max: float,
) -> np.ndarray:
    """Clip quantile levels to ``[q_min, q_max]`` for backend evaluation."""
    return np.clip(np.asarray(quantiles, dtype=np.float64), q_min, q_max)


def backend_quantile_levels(
    requested: Sequence[float],
    *,
    q_min: float,
    q_max: float,
    include_median: bool = True,
) -> list[float]:
    """Unique sorted quantile levels to request from a native quantile backend."""
    levels = {float(q) for q in clip_quantiles_to_range(requested, q_min, q_max)}
    if include_median:
        levels.add(float(np.clip(0.5, q_min, q_max)))
    return sorted(levels)


def select_clipped_quantile_values(
    backend_levels: Sequence[float],
    values: np.ndarray,
    requested: Sequence[float],
    *,
    q_min: float,
    q_max: float,
    axis: int = -1,
) -> np.ndarray:
    """Map backend forecasts to user-requested quantiles with edge clamping."""
    clipped = clip_quantiles_to_range(requested, q_min, q_max)
    indices = [_match_quantile_level(backend_levels, float(q)) for q in clipped]
    return np.take(values, indices, axis=axis)


def _knot_indices(
    knot_quantiles: np.ndarray,
    requested: Sequence[float],
) -> list[int] | None:
    indices: list[int] = []
    for q in requested:
        matches = np.where(np.isclose(knot_quantiles, q))[0]
        if len(matches) == 0:
            return None
        indices.append(int(matches[0]))
    return indices


def interpolate_quantiles(
    knot_quantiles: Sequence[float],
    knot_values: np.ndarray,
    requested_quantiles: Sequence[float],
    *,
    axis: int = -1,
) -> np.ndarray:
    """Linearly interpolate knot forecasts onto ``requested_quantiles``.

    Args:
        knot_quantiles: Native quantile levels predicted by the model, in
            ascending order.
        knot_values: Forecast array with native quantiles along ``axis``.
        requested_quantiles: Quantile levels to return.
        axis: Axis index of ``knot_quantiles`` in ``knot_values``.

    Returns:
        Array with the same shape as ``knot_values`` except ``axis`` is replaced
        by ``len(requested_quantiles)``.
    """
    knot_qs = np.asarray(knot_quantiles, dtype=np.float64)
    if knot_qs.ndim != 1 or len(knot_qs) < 1:
        raise ValueError("`knot_quantiles` must be a non-empty 1-D sequence.")
    if len(knot_qs) == 1:
        v = np.moveaxis(knot_values, axis, -1)
        out = np.repeat(v[..., :1], len(requested_quantiles), axis=-1)
        return np.moveaxis(out, -1, axis) if axis != -1 else out

    requested = np.clip(
        np.asarray(requested_quantiles, dtype=np.float64),
        knot_qs[0],
        knot_qs[-1],
    )
    values = np.moveaxis(knot_values, axis, -1)
    orig_shape = values.shape[:-1]
    flat = values.reshape(-1, len(knot_qs))
    interp = interp1d(
        knot_qs,
        flat,
        axis=1,
        kind="linear",
        assume_sorted=True,
    )
    out = interp(requested).reshape(*orig_shape, len(requested_quantiles))
    return np.moveaxis(out, -1, axis) if axis != -1 else out


def resolve_quantile_values(
    knot_quantiles: Sequence[float],
    knot_values: np.ndarray,
    requested_quantiles: Sequence[float],
    *,
    axis: int = -1,
) -> np.ndarray:
    """Select or interpolate ``knot_values`` onto ``requested_quantiles``."""
    knot_qs = np.asarray(knot_quantiles, dtype=np.float64)
    indices = _knot_indices(knot_qs, requested_quantiles)
    if indices is not None:
        return np.take(knot_values, indices, axis=axis)
    return interpolate_quantiles(
        knot_quantiles,
        knot_values,
        requested_quantiles,
        axis=axis,
    )
