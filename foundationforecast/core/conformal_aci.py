from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp

from .conformal import ConformalMethod


def _replay_aci(
    scores: np.ndarray,
    target_alpha: np.ndarray,
    *,
    step_size: int,
    final_gap: int,
    gamma: float,
    ncal: int,
    rolling: bool,
) -> np.ndarray:
    """Replay absolute-error ACI using only feedback revealed by each origin."""
    n_windows, n_series, horizon = scores.shape
    target = target_alpha.reshape(-1, 1)
    out = np.empty((target_alpha.size, n_series, horizon))
    final_origin = (n_windows - 1) * step_size + final_gap
    for j in range(horizon):
        lag = j + 1
        alpha = np.broadcast_to(target, (target_alpha.size, n_series)).copy()
        issued: dict[int, np.ndarray] = {}
        applied = ncal
        for m in range(ncal, n_windows + 1):
            origin = m * step_size if m < n_windows else final_origin
            while applied < n_windows and applied * step_size + lag <= origin:
                # Judge the score against its original interval, not today's one.
                miss = scores[applied, :, j][None, :] > issued[applied]
                alpha += gamma * (target - miss)
                applied += 1
            seen = min(n_windows, max(0, (origin - lag) // step_size + 1))
            start = max(0, seen - ncal) if rolling else 0
            if seen:
                ordered = np.sort(scores[start:seen, :, j], axis=0)
                # Clip lookup indices only; the adaptive alpha itself is not clipped.
                ranks = np.ceil(len(ordered) * (1 - alpha)).astype(np.intp) - 1
                indices = np.clip(ranks, 0, len(ordered) - 1)
                radius = np.take_along_axis(ordered, indices, axis=0)
            else:
                radius = np.full_like(alpha, np.inf)
            # The extended quantile gives the whole line / empty set at the edges.
            radius = np.where(alpha <= 0, np.inf, radius)
            radius = np.where(alpha >= 1, -np.inf, radius)
            issued[m] = radius
        out[:, :, j] = issued[n_windows]
    return out


@dataclass
class ACI(ConformalMethod):
    """Adaptive conformal inference reconstructed from calibration windows.

    Replays the Gibbs and Candes (2021) update
    ``alpha_next = alpha + gamma * (target_alpha - miss)`` separately for each
    series, horizon step, and level, starting at ``target_alpha = 1 - level/100``.
    Scores are absolute point-forecast errors. An interval uses the inverse
    empirical CDF of previously revealed scores, without interpolation or the
    finite-sample rank correction of ``conformal_error_finite_sample``.

    Args:
        gamma: Adaptation rate in ``(0, 1]`` (default ``0.01``). A miss decreases
            alpha and requests a higher quantile; a cover increases alpha.
        ncal: Initial windows used only to seed the score distribution (no alpha
            updates). ``None`` uses ``max(1, n_windows // 2)``. Must be smaller
            than ``n_windows``. A warm-up score enters the CDF when revealed,
            even if it was not yet observable at the first replay origin.
        rolling: Use only the latest ``ncal`` revealed scores for the CDF
            (``True``), or all revealed scores (``False``, default).

    Feedback at horizon step ``j`` is used only after ``j`` observations have
    passed since its forecast origin. Pending scores are checked against the
    intervals actually issued during replay, so overlapping windows do not
    leak future errors. No state persists across calls or models.

    Alpha is not clipped: ``alpha <= 0`` produces an unbounded interval and
    ``alpha >= 1`` an empty interval. Empty intervals are represented by lower
    ``+inf`` and upper ``-inf``. Output radii are widened across sorted levels
    to make intervals nested. Non-finite output bounds emit a ``UserWarning``.

    Resetting replay on each call, delayed feedback, and output nesting mean
    this API does not inherit the continuous online algorithm's long-run
    coverage guarantee. Measure coverage and width on held-out backtests.

    Example:
        ```python
        from foundationforecast import ACI, ConformalIntervals

        ConformalIntervals(n_windows=20, method=ACI(gamma=0.005))
        ```
    """

    gamma: float = 0.01
    ncal: int | None = None
    rolling: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.gamma, bool)
            or not math.isfinite(self.gamma)
            or not 0 < self.gamma <= 1
        ):
            raise ValueError("ACI gamma must be finite and in (0, 1]")
        if self.ncal is not None and (
            isinstance(self.ncal, bool)
            or not isinstance(self.ncal, Integral)
            or self.ncal < 1
        ):
            raise ValueError("ACI ncal must be a positive integer")

    def resolve_ncal(self, n_windows: int) -> int:
        return int(self.ncal) if self.ncal is not None else max(1, n_windows // 2)

    def validate(self, n_windows: int, step_size: int | None) -> None:
        if (
            isinstance(n_windows, bool)
            or not isinstance(n_windows, Integral)
            or n_windows < 2
        ):
            raise ValueError("ACI n_windows must be an integer of at least 2")
        if self.resolve_ncal(n_windows) >= n_windows:
            raise ValueError("ACI ncal must be smaller than n_windows to replay errors")
        if step_size is not None and (
            isinstance(step_size, bool)
            or not isinstance(step_size, Integral)
            or step_size < 1
        ):
            raise ValueError("ACI step_size must be a positive integer")

    def thresholds(
        self,
        errors: np.ndarray,
        level: Sequence[int | float],
        step_size: int,
        final_gap: int,
    ) -> np.ndarray:
        """Return raw symmetric radii of shape ``(n_levels, n_series, h)``.

        Errors have shape ``(n_windows, n_series, h)`` and are oldest first.
        ``final_gap`` is the distance from the last calibration origin to the
        forecast origin; ``forecast()`` passes its calibration horizon here.
        This mathematical helper does not nest radii or warn about edge sets.
        """
        errors = np.asarray(errors, dtype=float)
        if errors.ndim != 3 or min(errors.shape) < 1:
            raise ValueError("ACI errors must have shape (n_windows, n_series, h)")
        self.validate(errors.shape[0], step_size)
        if (
            isinstance(final_gap, bool)
            or not isinstance(final_gap, Integral)
            or final_gap < 1
        ):
            raise ValueError("ACI final_gap must be a positive integer")
        if not np.isfinite(errors).all():
            raise ValueError("ACI requires finite calibration errors")
        levels = np.asarray(level, dtype=float)
        if (
            levels.ndim != 1
            or levels.size == 0
            or not np.isfinite(levels).all()
            or np.any((levels < 0) | (levels > 100))
        ):
            raise ValueError("ACI levels must be finite and in [0, 100]")
        return _replay_aci(
            np.abs(errors),
            1 - levels / 100,
            step_size=step_size,
            final_gap=final_gap,
            gamma=self.gamma,
            ncal=self.resolve_ncal(errors.shape[0]),
            rolling=self.rolling,
        )

    def __call__(
        self,
        fcst_df: pd.DataFrame,
        cs_df: pd.DataFrame,
        model_names: list[str],
        level: list[int | float],
        cs_n_windows: int,
        cs_h: int,
        n_series: int,
        horizon: int,
        step_size: int,
    ) -> pd.DataFrame:
        fcst_df = ufp.copy_if_pandas(fcst_df, deep=False)
        unbounded = np.zeros(len(level), dtype=bool)
        empty = np.zeros(len(level), dtype=bool)
        for model in model_names:
            mean = fcst_df[model].to_numpy().reshape(1, n_series, horizon)
            errors = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
            raw = self.thresholds(errors[:, :, :horizon], level, step_size, cs_h)
            radius = np.maximum.accumulate(raw, axis=0)
            unbounded |= np.isposinf(radius).any(axis=(1, 2))
            empty |= np.isneginf(radius).any(axis=(1, 2))
            bounds = np.vstack([mean - radius[::-1], mean + radius]).reshape(
                2 * len(level), -1
            )
            columns = [f"{model}-lo-{lv}" for lv in reversed(level)]
            columns += [f"{model}-hi-{lv}" for lv in level]
            fcst_df = ufp.assign_columns(fcst_df, columns, bounds.T)
        descriptions = []
        for mask, kind in [(unbounded, "unbounded"), (empty, "empty")]:
            if mask.any():
                affected = ", ".join(
                    f"{lv:g}" for lv, bad in zip(level, mask, strict=False) if bad
                )
                descriptions.append(f"{kind} intervals at level(s) {affected}")
        if descriptions:
            warnings.warn(
                "ACI replay produced "
                + "; ".join(descriptions)
                + f" (n_windows={cs_n_windows}, "
                + f"ncal={self.resolve_ncal(cs_n_windows)}, gamma={self.gamma:g}). "
                "Unbounded bounds are -inf/+inf; empty "
                "bounds are +inf/-inf. Use more calibration windows or review gamma "
                "before plotting or scoring these intervals.",
                UserWarning,
                stacklevel=2,
            )
        return fcst_df
