from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .covariates import (
    XReg,
    exog_strategy_disabled,
    exog_strategy_requires_xreg,
    normalize_exog_strategy,
    resolve_futr_exog_list,
    resolve_horizon_exog_df,
    sort_exog_panel,
    validate_exog_strategy_for_timegpt,
    validate_futr_exog_inputs,
)
from .xreg import merge_xreg_into_forecast_df

if TYPE_CHECKING:
    from ..forecaster import Forecaster


@dataclass(frozen=True)
class FutrExogContext:
    horizon_df: pd.DataFrame
    futr_exog_list: list[str]
    use_xreg: bool
    xreg: XReg


def prepare_futr_exog_context(
    forecaster: Forecaster,
    df: pd.DataFrame,
    h: int,
    *,
    X_df: pd.DataFrame | None,
    futr_df: pd.DataFrame | None,
    futr_exog_list: list[str] | None,
) -> FutrExogContext | None:
    strategy = normalize_exog_strategy(getattr(forecaster, "exog_strategy", "auto"))
    if exog_strategy_disabled(strategy):
        return None

    horizon_df = resolve_horizon_exog_df(X_df, futr_df)
    if horizon_df is None:
        if futr_exog_list is not None:
            raise ValueError("futr_exog_list was provided without X_df or futr_df.")
        return None

    cols = resolve_futr_exog_list(horizon_df, futr_exog_list)
    validate_futr_exog_inputs(df, h, horizon_df, cols)
    horizon_df = sort_exog_panel(horizon_df)

    if type(forecaster).__name__ == "TimeGPT":
        validate_exog_strategy_for_timegpt(strategy)  # type: ignore[arg-type]

    supports_native = forecaster.supports_native_futr_exog()
    if strategy == "native" and not supports_native:
        raise ValueError(
            f"{type(forecaster).__name__} does not support native future-known "
            "exogenous variables. Use exog_strategy='auto' or XReg(...)."
        )

    use_xreg = exog_strategy_requires_xreg(
        strategy,  # type: ignore[arg-type]
        supports_native=supports_native,
    )
    xreg = strategy if isinstance(strategy, XReg) else XReg()
    return FutrExogContext(
        horizon_df=horizon_df,
        futr_exog_list=cols,
        use_xreg=use_xreg,
        xreg=xreg,
    )


def dispatch_futr_exog_forecast(
    forecaster: Forecaster,
    ctx: FutrExogContext,
    df: pd.DataFrame,
    h: int,
    freq: str | None,
    level: list[int | float] | None,
    quantiles: list[float] | None,
    panel,
    univariate_forecast: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    if ctx.use_xreg:
        fcst = univariate_forecast()
        return merge_xreg_into_forecast_df(
            fcst,
            forecaster.alias,
            df,
            ctx.horizon_df,
            ctx.futr_exog_list,
            h,
            ctx.xreg,
        )
    native = forecaster._forecast_native_futr_exog(
        df=df,
        h=h,
        freq=freq,
        level=level,
        quantiles=quantiles,
        panel=panel,
        horizon_df=ctx.horizon_df,
        futr_exog_list=ctx.futr_exog_list,
    )
    if native is None:
        raise RuntimeError("Native futr exog path returned None unexpectedly.")
    return native
