from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

ID_COL = "unique_id"
TIME_COL = "ds"
TARGET_COL = "y"
META_COLS = frozenset({ID_COL, TIME_COL, TARGET_COL})

ExogStrategyName: TypeAlias = Literal["auto", "native"]


@dataclass
class XReg:
    """TimesFM-style FM + regressor decomposition for known-future exogenous variables.

    Exogenous variables (exog) are covariates known over the forecast horizon.
    """

    fm_first: bool = True
    regressor: Literal["linear"] = "linear"


ExogStrategyConfig: TypeAlias = ExogStrategyName | XReg | None

_VALID_EXOG_STRATEGY_NAMES = frozenset({"auto", "native"})


def normalize_exog_strategy(
    exog_strategy: ExogStrategyConfig,
) -> ExogStrategyName | XReg:
    if exog_strategy is None:
        return "auto"
    if (
        isinstance(exog_strategy, str)
        and exog_strategy not in _VALID_EXOG_STRATEGY_NAMES
    ):
        raise ValueError(
            f"Invalid exog_strategy {exog_strategy!r}. "
            f"Use one of {sorted(_VALID_EXOG_STRATEGY_NAMES)} or XReg(...)."
        )
    return exog_strategy


def resolve_horizon_exog_df(
    X_df: pd.DataFrame | None,
    futr_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if X_df is not None and futr_df is not None and X_df is not futr_df:
        raise ValueError(
            "X_df and futr_df must be the same DataFrame when both are provided."
        )
    return X_df if X_df is not None else futr_df


def infer_futr_exog_columns(horizon_df: pd.DataFrame) -> list[str]:
    return [c for c in horizon_df.columns if c not in META_COLS]


def resolve_futr_exog_list(
    horizon_df: pd.DataFrame,
    futr_exog_list: list[str] | None,
) -> list[str]:
    if futr_exog_list is not None:
        return list(futr_exog_list)
    cols = infer_futr_exog_columns(horizon_df)
    if not cols:
        raise ValueError(
            "No exogenous columns found in X_df. Pass futr_exog_list or include "
            "covariate columns in X_df (excluding unique_id, ds, y)."
        )
    return cols


def validate_futr_exog_inputs(
    df: pd.DataFrame,
    h: int,
    horizon_df: pd.DataFrame,
    futr_exog_list: list[str],
) -> None:
    if ID_COL not in horizon_df.columns or TIME_COL not in horizon_df.columns:
        raise ValueError(f"X_df must include columns '{ID_COL}' and '{TIME_COL}'.")

    missing_hist = [c for c in futr_exog_list if c not in df.columns]
    if missing_hist:
        raise ValueError(
            f"Exogenous columns missing from history df: {missing_hist}. "
            "Known-future covariates must appear in df for the observed period."
        )

    missing_hor = [c for c in futr_exog_list if c not in horizon_df.columns]
    if missing_hor:
        raise ValueError(f"Exogenous columns missing from X_df: {missing_hor}.")

    hist_ids = set(df[ID_COL].unique())
    hor_ids = set(horizon_df[ID_COL].unique())
    extra_ids = sorted(hor_ids - hist_ids)
    if extra_ids:
        raise ValueError(
            f"X_df contains unique_id values not present in df: {extra_ids}"
        )

    counts = horizon_df.groupby(ID_COL, observed=True).size()
    bad = counts[counts != h]
    if len(bad) > 0:
        raise ValueError(
            f"X_df must have exactly h={h} rows per unique_id. "
            f"Offending counts: {bad.to_dict()}"
        )


def validate_exog_strategy_for_timegpt(exog_strategy: ExogStrategyName | XReg) -> None:
    if isinstance(exog_strategy, XReg):
        raise ValueError(
            "TimeGPT does not support exog_strategy=XReg(...). "
            "Use exog_strategy='auto' or 'native' with X_df at forecast time."
        )


def exog_strategy_requires_xreg(
    exog_strategy: ExogStrategyName | XReg,
    *,
    supports_native: bool,
) -> bool:
    if isinstance(exog_strategy, XReg):
        return True
    if exog_strategy == "native":
        return False
    return not supports_native
