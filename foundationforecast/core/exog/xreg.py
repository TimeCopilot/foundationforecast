from __future__ import annotations

import numpy as np
import pandas as pd

from .covariates import TARGET_COL, XReg

_EXOG_STDDEV_EPS = 1e-8


def _standardize_exog_for_regression(
    exog_hist: np.ndarray,
    exog_horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score exog columns using in-sample mean and std (per column)."""
    mean = np.mean(exog_hist, axis=0)
    std = np.std(exog_hist, axis=0, ddof=0)
    std = np.where(std < _EXOG_STDDEV_EPS, 1.0, std)
    return (exog_hist - mean) / std, (exog_horizon - mean) / std


def _fit_linear(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return coefficients beta for y ~ x (no intercept)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = x.astype(np.float64)
    coef, _, _, _ = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)
    return coef


def _predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.astype(np.float64) @ coef


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

    return _adjust_uid_forecasts_with_xreg(
        alias="__point__",
        y_hist=y_hist,
        exog_hist=exog_hist,
        exog_horizon=exog_horizon,
        baselines={"__point__": baseline_horizon},
        xreg=xreg,
    )["__point__"]


def _forecast_value_columns(fcst_df: pd.DataFrame, alias: str) -> list[str]:
    """Point and probabilistic forecast columns produced for ``alias``."""
    cols = [alias]
    prefix = f"{alias}-"
    for col in fcst_df.columns:
        if col.startswith(prefix):
            cols.append(col)
    return cols


def _adjust_uid_forecasts_with_xreg(
    *,
    alias: str,
    y_hist: np.ndarray,
    exog_hist: np.ndarray,
    exog_horizon: np.ndarray,
    baselines: dict[str, np.ndarray],
    xreg: XReg,
) -> dict[str, np.ndarray]:
    """Apply XReg to point and interval/quantile baselines for one series."""
    if xreg.regressor != "linear":
        raise NotImplementedError(f"Unsupported XReg regressor: {xreg.regressor!r}")

    exog_hist, exog_horizon = _standardize_exog_for_regression(exog_hist, exog_horizon)
    point_horizon = baselines[alias]

    if xreg.fm_first:
        baseline_insample = np.full(len(y_hist), float(point_horizon[0]))
        coef = _fit_linear(y_hist - baseline_insample, exog_hist)
        adjustment = _predict_linear(coef, exog_horizon)
        return {col: baseline + adjustment for col, baseline in baselines.items()}

    coef = _fit_linear(y_hist, exog_hist)
    reg_hist = _predict_linear(coef, exog_hist)
    reg_hor = _predict_linear(coef, exog_horizon)
    mean_resid = float(np.mean(y_hist - reg_hist))
    return {
        col: reg_hor + (baseline - mean_resid) for col, baseline in baselines.items()
    }


def merge_xreg_into_forecast_df(
    fcst_df: pd.DataFrame,
    alias: str,
    df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    futr_exog_list: list[str],
    h: int,
    xreg: XReg,
) -> pd.DataFrame:
    out = fcst_df.copy()
    value_cols = _forecast_value_columns(out, alias)
    if alias not in value_cols:
        raise ValueError(f"Forecast frame missing point column {alias!r}.")

    adjusted_by_col: dict[str, list[np.ndarray]] = {col: [] for col in value_cols}

    for uid in out["unique_id"].unique():
        n = h
        uid_mask = out["unique_id"] == uid
        y_hist = df.loc[df["unique_id"] == uid, TARGET_COL].to_numpy(dtype=np.float64)
        exog_hist = df.loc[df["unique_id"] == uid, futr_exog_list].to_numpy(
            dtype=np.float64
        )
        exog_hor = horizon_df.loc[
            horizon_df["unique_id"] == uid, futr_exog_list
        ].to_numpy(dtype=np.float64)
        if len(y_hist) != len(exog_hist):
            raise ValueError("History length mismatch for exogenous columns.")
        if exog_hor.shape[0] != n:
            raise ValueError("Horizon exog length mismatch.")

        baselines = {
            col: out.loc[uid_mask, col].to_numpy(dtype=np.float64) for col in value_cols
        }
        adjusted = _adjust_uid_forecasts_with_xreg(
            alias=alias,
            y_hist=y_hist,
            exog_hist=exog_hist,
            exog_horizon=exog_hor,
            baselines=baselines,
            xreg=xreg,
        )
        for col in value_cols:
            adjusted_by_col[col].append(adjusted[col])

    for col in value_cols:
        out[col] = np.concatenate(adjusted_by_col[col])
    return out
