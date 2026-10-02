from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .covariates import (
    exog_strategy_disabled,
    normalize_exog_strategy,
    resolve_futr_exog_list,
    resolve_horizon_exog_df,
    sort_exog_panel,
    validate_futr_exog_inputs,
)

if TYPE_CHECKING:
    from ..forecaster import Forecaster

logger = logging.getLogger(__name__)

_FUTR_EXOG_DOC = "docs/exogenous-variables.md"


def futr_exog_unsupported_message(forecaster: Forecaster) -> str:
    name = type(forecaster).__name__
    repo_id = getattr(forecaster, "repo_id", None)
    if repo_id is not None:
        name = f"{name} (repo_id={repo_id!r})"
    return (
        f"{name} does not support known-future exogenous variables for this "
        f"checkpoint. See {_FUTR_EXOG_DOC} for native checkpoints, or set "
        "exog_strategy=False to ignore X_df when using FoundationForecast."
    )


@dataclass(frozen=True)
class FutrExogContext:
    horizon_df: pd.DataFrame
    futr_exog_list: list[str]


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

    if not forecaster.supports_native_futr_exog():
        msg = futr_exog_unsupported_message(forecaster)
        logger.warning(msg)
        raise ValueError(msg)

    return FutrExogContext(
        horizon_df=horizon_df,
        futr_exog_list=cols,
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
    _ = univariate_forecast
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
