from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp

from .conformal import ConformalMethod


def _saturated_tan(x: np.ndarray) -> np.ndarray:
    """``tan(x)`` with ``±inf`` outside ``[-pi/2, pi/2]``."""
    half_pi = math.pi / 2
    with np.errstate(invalid="ignore"):
        inner = np.tan(np.clip(x, -half_pi, half_pi))
    return np.where(x >= half_pi, np.inf, np.where(x <= -half_pi, -np.inf, inner))


def default_csat(tg: float, delta: float) -> float:
    """Saturation constant that targets coverage ``1 - alpha - delta`` by ``tg``."""
    csat = 2 / math.pi * (math.ceil(math.log(tg) * delta) - 1 / math.log(tg))
    if csat <= 0:
        raise ValueError(
            f"Tg={tg:g} and delta={delta:g} give a non-positive Csat; "
            "increase Tg or pass Csat explicitly"
        )
    return csat


def track_quantiles(
    scores: np.ndarray,
    alpha: np.ndarray,
    *,
    step_size: int,
    final_gap: int,
    lr: float,
    ki: np.ndarray,
    csat: float,
    ncal: int,
    rolling: bool,
    integrate: bool,
    warm_start: bool,
) -> np.ndarray:
    """Replay quantile tracking plus error integration on one score tail.

    Args:
        scores: ``(n_windows, n_series, h)`` nonconformity scores, oldest
            window first. Window ``k`` forecast origin is ``k * step_size``.
        alpha: ``(n_levels,)`` target miss rate of this tail.
        step_size: Origin spacing between consecutive windows.
        final_gap: Distance between the last window origin and the origin of the
            forecast being calibrated (the cross-validation horizon).
        lr: Base learning rate; scaled by the recent score range.
        ki: ``(n_series,)`` integrator scale.
        csat: Integrator saturation constant.
        ncal: Window length for the rolling learning-rate range, and the number
            of initial windows used for the warm start.
        rolling: Use only the last ``ncal`` scores for the learning-rate range.
        integrate: Include the error-integration term.
        warm_start: Start the tracker from the empirical quantile of the first
            ``ncal`` windows instead of zero.

    Returns:
        ``(n_levels, n_series, h)`` thresholds for the next forecast origin.

    The score of window ``k`` at horizon ``j`` is only revealed ``j`` steps
    after its origin, so each decision only uses windows whose scores had been
    observed by then (``k * step_size + j <= origin``). The final forecast
    sits ``final_gap`` after the last window and therefore sees every score.
    """
    n_windows, n_series, horizon = scores.shape
    n_levels = alpha.size
    a = alpha.reshape(n_levels, 1)
    first = ncal if warm_start else 0
    last_origin = (n_windows - 1) * step_size + final_gap
    running_max = np.maximum.accumulate(scores, axis=0)
    running_min = np.minimum.accumulate(scores, axis=0)
    out = np.empty((n_levels, n_series, horizon))
    for j in range(horizon):
        lag = j + 1
        s = scores[:, :, j]
        p = np.zeros((n_levels, n_series))
        if warm_start:
            seen = [k for k in range(first) if k * step_size + lag <= first * step_size]
            if seen:
                p = np.quantile(s[seen], 1 - alpha, axis=0)
        integral = np.zeros((n_levels, n_series))
        misses = np.zeros((n_levels, n_series))
        n_obs = 0
        issued: dict[int, np.ndarray] = {}
        applied = first
        for m in range(first, n_windows + 1):
            origin = m * step_size if m < n_windows else last_origin
            while applied < n_windows and applied * step_size + lag <= origin:
                k = applied
                miss = (s[k][None, :] > issued[k]).astype(float)
                start = max(0, k - ncal + 1) if rolling else 0
                if k - start + 1 <= 1:
                    eta: np.ndarray | float = lr
                elif rolling:
                    window = s[start : k + 1]
                    eta = lr * (window.max(axis=0) - window.min(axis=0))
                else:
                    eta = lr * (running_max[k, :, j] - running_min[k, :, j])
                p = p + np.multiply(eta, miss - a)
                misses += miss
                n_obs += 1
                if integrate and n_obs > 1:
                    arg = misses - n_obs * a
                    sat = _saturated_tan(arg * math.log(n_obs) / (csat * n_obs))
                    with np.errstate(invalid="ignore"):
                        integral = np.where(ki > 0, ki * sat, 0.0)
                applied += 1
            issued[m] = p + integral
        out[:, :, j] = issued[n_windows]
    return out


@dataclass
class PID(ConformalMethod):
    """Conformal PID control for online interval calibration under shift.

    Implements the proportional and integral modules of Angelopoulos, Candes
    and Tibshirani (2023). Starting from the oldest calibration window the
    method replays the backtest in time order. Per horizon it tracks the error
    quantile directly on the score scale:

    - **P** (quantile tracking) moves the threshold up after a miss and down
      after a cover, by ``lr`` times the recent range of the errors.
    - **I** (error integration) feeds the cumulative miscoverage through a
      saturating ``tan`` so that persistent coverage bias is corrected.

    Errors are delay-aware: the error at horizon ``h`` of a window is only used
    ``h`` observations after its forecast origin, so ``step_size < h`` windows
    never see the future. The forecast being calibrated sees every window.

    Args:
        symmetric: ``False`` (default) tracks the lower and upper tails
            separately (each targets ``alpha / 2`` misses), so intervals can be
            asymmetric. ``True`` tracks the absolute error.
        lr: Base learning rate of quantile tracking, scaled internally by the
            range of recent errors.
        integrate: Include the error-integration term.
        KI: Integrator scale. ``None`` uses the largest absolute calibration
            error of each series.
        Csat: Integrator saturation constant. ``None`` derives it from ``Tg``
            and ``delta`` so the absolute coverage is at least
            ``1 - alpha - delta`` by time ``Tg``.
        Tg: Target time for the coverage guarantee. ``None`` uses
            ``n_windows``.
        delta: Tolerated absolute coverage gap in ``(0, 1)``.
        ncal: Initial windows used to warm start the tracker, and the trailing
            window of the learning-rate range when ``rolling``. ``None`` uses
            ``max(1, n_windows // 2)``.
        rolling: Use only the last ``ncal`` errors for the learning-rate range
            instead of every error observed so far.
        init: ``"quantile"`` starts each tracker from the empirical quantile of
            the first ``ncal`` windows and replays the rest, which works with
            the few backtest windows typical for foundation models.
            ``"zero"`` starts from zero and replays every window (the
            reference formulation; needs many windows to warm up).

    Example:
        ```python
        from foundationforecast import ConformalIntervals, PID

        ConformalIntervals(n_windows=20, method=PID(lr=0.1))
        ```

    Each level is tracked independently. Thresholds are then floored at zero
    (intervals always contain the point forecast) and sorted across levels so
    wider levels contain narrower ones. When the integrator saturates (the
    realized miss rate is far from the target) the bound becomes infinite and
    ``forecast()`` emits a ``UserWarning``. The scorecasting (D) module is not
    implemented.
    """

    symmetric: bool = False
    lr: float = 0.1
    integrate: bool = True
    KI: float | None = None
    Csat: float | None = None
    Tg: float | None = None
    delta: float = 0.01
    ncal: int | None = None
    rolling: bool = False
    init: Literal["quantile", "zero"] = "quantile"

    def __post_init__(self) -> None:
        if not self.lr > 0:
            raise ValueError("lr must be positive")
        if self.KI is not None and not self.KI >= 0:
            raise ValueError("KI must be non-negative")
        if self.Csat is not None and not self.Csat > 0:
            raise ValueError("Csat must be positive")
        if self.Tg is not None and not self.Tg > 1:
            raise ValueError("Tg must be greater than 1")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.ncal is not None and self.ncal < 1:
            raise ValueError("ncal must be a positive integer")
        if self.init not in ("quantile", "zero"):
            raise ValueError("init must be 'quantile' or 'zero'")

    def resolve_ncal(self, n_windows: int) -> int:
        return self.ncal if self.ncal is not None else max(1, n_windows // 2)

    def validate(self, n_windows: int, step_size: int | None) -> None:
        if self.init == "quantile" and self.resolve_ncal(n_windows) >= n_windows:
            raise ValueError(
                f"PID ncal={self.resolve_ncal(n_windows)} must be smaller than "
                f"n_windows={n_windows} so some windows remain to be replayed"
            )

    def thresholds(
        self,
        errors: np.ndarray,
        level: list[int | float],
        step_size: int,
        final_gap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper thresholds ``(n_levels, n_series, h)`` for ``errors``.

        ``errors`` are signed ``y - y_hat`` with shape
        ``(n_windows, n_series, h)``, oldest window first. The interval is
        ``[y_hat - lower, y_hat + upper]``.
        """
        n_windows = errors.shape[0]
        self.validate(n_windows, step_size)
        alpha = 1 - np.asarray(level, dtype=float) / 100
        tg = self.Tg if self.Tg is not None else float(n_windows)
        csat = self.Csat if self.Csat is not None else default_csat(tg, self.delta)
        ki = (
            np.full(errors.shape[1], self.KI, dtype=float)
            if self.KI is not None
            else np.abs(errors).max(axis=(0, 2))
        )
        kwargs: dict[str, Any] = dict(
            step_size=step_size,
            final_gap=final_gap,
            lr=self.lr,
            ki=ki,
            csat=csat,
            ncal=self.resolve_ncal(n_windows),
            rolling=self.rolling,
            integrate=self.integrate,
            warm_start=self.init == "quantile",
        )
        if self.symmetric:
            radius = track_quantiles(np.abs(errors), alpha, **kwargs)
            return radius, radius
        lower = track_quantiles(-errors, alpha / 2, **kwargs)
        upper = track_quantiles(errors, alpha / 2, **kwargs)
        return lower, upper

    def radii(
        self,
        errors: np.ndarray,
        level: list[int | float],
        step_size: int,
        final_gap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`thresholds` floored at zero and nested across ``level``.

        ``level`` must be sorted ascending. Warns when a bound is infinite.
        """
        lower, upper = self.thresholds(errors, level, step_size, final_gap)
        lower = np.maximum.accumulate(np.maximum(lower, 0.0), axis=0)
        upper = np.maximum.accumulate(np.maximum(upper, 0.0), axis=0)
        if np.isinf(lower).any() or np.isinf(upper).any():
            warnings.warn(
                "PID error integration saturated: some bounds are infinite. "
                "Use more calibration windows, a larger Csat or Tg, or "
                "integrate=False.",
                UserWarning,
                stacklevel=3,
            )
        return lower, upper

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
        lo_cols_suffix = list(reversed(level))
        for model in model_names:
            mean = fcst_df[model].to_numpy().reshape(1, n_series, -1)
            errors = cs_df[model].to_numpy().reshape(cs_n_windows, n_series, cs_h)
            lower, upper = self.radii(
                errors[:, :, :horizon], level, step_size, final_gap=cs_h
            )
            bounds = np.vstack([mean - lower[::-1], mean + upper]).reshape(
                2 * len(level), -1
            )
            lo_cols = [f"{model}-lo-{lv}" for lv in lo_cols_suffix]
            hi_cols = [f"{model}-hi-{lv}" for lv in level]
            fcst_df = ufp.assign_columns(fcst_df, lo_cols + hi_cols, bounds.T)
        return fcst_df
