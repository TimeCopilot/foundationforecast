from __future__ import annotations

import numpy as np
import pandas as pd

from .covariates import TARGET_COL, XReg


def _fit_linear(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return coefficients [intercept, beta...] for y ~ [1, x]."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), x.astype(np.float64)])
    coef, _, _, _ = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)
    return coef


def _predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(x)), x.astype(np.float64)])
    return design @ coef


def adjust_point_forecast_with_xreg(
    *,
    y_hist: np.ndarray,
    exog_hist: np.ndarray,
    exog_horizon: np.ndarray,
    baseline_horizon: np.ndarray,
    xreg: XReg,
) -> np.ndarray:
    """Apply linear XReg adjustment to a univariate baseline horizon forecast."""
    y_hist = np.asarray(y_hist, dtype=np.float64)
    exog_hist = np.asarray(exog_hist, dtype=np.float64)
    exog_horizon = np.asarray(exog_horizon, dtype=np.float64)
    baseline_horizon = np.asarray(baseline_horizon, dtype=np.float64)

    if xreg.regressor != "linear":
        raise NotImplementedError(f"Unsupported XReg regressor: {xreg.regressor!r}")

    if xreg.fm_first:
        baseline_insample = np.full(len(y_hist), float(baseline_horizon[0]))
        coef = _fit_linear(y_hist - baseline_insample, exog_hist)
        adjustment = _predict_linear(coef, exog_horizon)
        return baseline_horizon + adjustment

    coef = _fit_linear(y_hist, exog_hist)
    reg_hist = _predict_linear(coef, exog_hist)
    reg_hor = _predict_linear(coef, exog_horizon)
    residual_hist = y_hist - reg_hist
    residual_baseline = baseline_horizon - float(np.mean(residual_hist))
    return reg_hor + residual_baseline


def _has_probabilistic_columns(fcst_df: pd.DataFrame, alias: str) -> bool:
    meta = {"unique_id", "ds"}
    for col in fcst_df.columns:
        if col in meta or col == alias:
            continue
        if col.startswith(f"{alias}-"):
            return True
    return False


def merge_xreg_into_forecast_df(
    fcst_df: pd.DataFrame,
    alias: str,
    df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    futr_exog_list: list[str],
    h: int,
    xreg: XReg,
) -> pd.DataFrame:
    if _has_probabilistic_columns(fcst_df, alias):
        raise ValueError(
            "XReg linear fallback only supports point forecasts. "
            "Use level=None and quantiles=None, or a model with native exog support."
        )
    out = fcst_df.copy()
    baseline = out[alias].to_numpy(dtype=np.float64)
    adjusted_parts: list[np.ndarray] = []
    offset = 0
    for uid in out["unique_id"].unique():
        n = h
        y_hist = df.loc[df["unique_id"] == uid, TARGET_COL].to_numpy(dtype=np.float64)
        exog_hist = df.loc[df["unique_id"] == uid, futr_exog_list].to_numpy(
            dtype=np.float64
        )
        exog_hor = horizon_df.loc[
            horizon_df["unique_id"] == uid, futr_exog_list
        ].to_numpy(dtype=np.float64)
        base_hor = baseline[offset : offset + n]
        if len(y_hist) != len(exog_hist):
            raise ValueError("History length mismatch for exogenous columns.")
        if exog_hor.shape[0] != n:
            raise ValueError("Horizon exog length mismatch.")
        adjusted = adjust_point_forecast_with_xreg(
            y_hist=y_hist,
            exog_hist=exog_hist,
            exog_horizon=exog_hor,
            baseline_horizon=base_hor,
            xreg=xreg,
        )
        adjusted_parts.append(adjusted)
        offset += n
    out[alias] = np.concatenate(adjusted_parts)
    return out
