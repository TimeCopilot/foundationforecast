from __future__ import annotations

from typing import Literal, TypeAlias

import pandas as pd

ID_COL = "unique_id"
TIME_COL = "ds"
TARGET_COL = "y"
META_COLS = frozenset({ID_COL, TIME_COL, TARGET_COL})

ExogStrategyName: TypeAlias = Literal["auto"]

ExogStrategyConfig: TypeAlias = ExogStrategyName | None | Literal[False]

NormalizedExogStrategy: TypeAlias = ExogStrategyName | Literal[False]

_VALID_EXOG_STRATEGY_NAMES = frozenset({"auto", "native"})


def exog_strategy_disabled(exog_strategy: NormalizedExogStrategy) -> bool:
    return exog_strategy is False


def normalize_exog_strategy(
    exog_strategy: ExogStrategyConfig,
) -> NormalizedExogStrategy:
    if exog_strategy is None:
        return "auto"
    if exog_strategy is False:
        return False
    if not isinstance(exog_strategy, str):
        raise ValueError(
            f"Invalid exog_strategy type {type(exog_strategy)!r}. "
            f"Use one of {sorted(_VALID_EXOG_STRATEGY_NAMES)} or False."
        )
    if exog_strategy not in _VALID_EXOG_STRATEGY_NAMES:
        raise ValueError(
            f"Invalid exog_strategy {exog_strategy!r}. "
            f"Use one of {sorted(_VALID_EXOG_STRATEGY_NAMES)} or False."
        )
    if exog_strategy == "native":
        return "auto"
    return exog_strategy


def sort_exog_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Sort panel rows once for exog paths (``unique_id``, then ``ds``)."""
    if df.empty:
        return df
    return df.sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)


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


def _validate_futr_exog_column_names(cols: list[str]) -> list[str]:
    if not cols:
        raise ValueError("futr_exog_list must name at least one exogenous column.")
    reserved = [c for c in cols if c in META_COLS]
    if reserved:
        raise ValueError(
            f"futr_exog_list must not include metadata columns "
            f"{sorted(META_COLS)}; got {reserved}."
        )
    if len(cols) != len(set(cols)):
        raise ValueError("futr_exog_list must not contain duplicate names.")
    return list(cols)


def resolve_exog_columns_from_df(
    df: pd.DataFrame,
    futr_exog_list: list[str] | None,
) -> list[str]:
    """Column names for known-future exog stored in ``df`` (history + horizon)."""
    if futr_exog_list is not None:
        cols = _validate_futr_exog_column_names(futr_exog_list)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Exogenous columns missing from df: {missing}. "
                "Include them through the end of each cross-validation fold."
            )
        return cols
    return [c for c in df.columns if c not in META_COLS]


def resolve_futr_exog_list(
    horizon_df: pd.DataFrame,
    futr_exog_list: list[str] | None,
) -> list[str]:
    if futr_exog_list is not None:
        cols = _validate_futr_exog_column_names(futr_exog_list)
        missing = [c for c in cols if c not in horizon_df.columns]
        if missing:
            raise ValueError(f"Exogenous columns missing from X_df: {missing}.")
        return cols
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
    missing_ids = sorted(hist_ids - hor_ids)
    if missing_ids:
        raise ValueError(
            f"X_df is missing unique_id values present in df: {missing_ids}"
        )

    if horizon_df.duplicated(subset=[ID_COL, TIME_COL]).any():
        raise ValueError(
            f"X_df must not contain duplicate ({ID_COL}, {TIME_COL}) rows."
        )

    counts = horizon_df.groupby(ID_COL, observed=True).size()
    bad = counts[counts != h]
    if len(bad) > 0:
        raise ValueError(
            f"X_df must have exactly h={h} rows per unique_id. "
            f"Offending counts: {bad.to_dict()}"
        )
