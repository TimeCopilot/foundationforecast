from __future__ import annotations

from typing import TYPE_CHECKING

import fev
import pandas as pd
from fev import convert_input_data
from fev.constants import PREDICTIONS

from .jobs import is_known_dynamic_only
from foundationforecast.core.quantiles import quantile_column_name

if TYPE_CHECKING:
    from foundationforecast.core.forecaster import Forecaster


def _fcst_df_to_window_predictions(
    fcst_df: pd.DataFrame,
    *,
    alias: str,
    quantile_levels: list[float],
    horizon: int,
) -> list[dict]:
    if fcst_df.empty:
        raise ValueError("Empty forecast dataframe")
    required = {"unique_id", "ds", alias}
    missing = required - set(fcst_df.columns)
    if missing:
        raise ValueError(f"Forecast missing columns {sorted(missing)}")

    preds: list[dict] = []
    for _uid, group in fcst_df.groupby("unique_id", sort=True):
        group = group.sort_values("ds")
        if len(group) != horizon:
            raise ValueError(
                f"Expected {horizon} forecast rows for series, got {len(group)}"
            )
        row: dict = {
            PREDICTIONS: group[alias].astype(float).tolist(),
        }
        for q in quantile_levels:
            col = quantile_column_name(alias, q)
            if col not in group.columns:
                raise ValueError(f"Missing quantile column {col!r} in forecast output")
            row[str(q)] = group[col].astype(float).tolist()
        preds.append(row)
    return preds


def forecast_task(
    task: fev.Task,
    forecaster: Forecaster,
    *,
    num_proc: int = 8,
) -> list[list[dict]]:
    if not is_known_dynamic_only(task):
        raise ValueError(
            f"Task {task.task_name!r} is not known-dynamic-only; "
            "see configs/known_only_tasks.yaml"
        )
    if not task.known_dynamic_columns:
        raise ValueError(f"Task {task.task_name!r} has no known_dynamic_columns")

    alias = forecaster.alias
    predictions_per_window: list[list[dict]] = []

    for window in task.iter_windows(num_proc=num_proc):
        past_df, future_df, _static_df = convert_input_data(window, adapter="nixtla")
        futr_exog_list = list(task.known_dynamic_columns)
        X_df = future_df[["unique_id", "ds", *futr_exog_list]]

        fcst_df = forecaster.forecast(
            df=past_df,
            h=task.horizon,
            freq=task.freq,
            quantiles=task.quantile_levels,
            X_df=X_df,
            futr_exog_list=futr_exog_list,
        )
        predictions_per_window.append(
            _fcst_df_to_window_predictions(
                fcst_df,
                alias=alias,
                quantile_levels=task.quantile_levels,
                horizon=task.horizon,
            )
        )

    if len(predictions_per_window) != task.num_windows:
        raise RuntimeError(
            f"Expected {task.num_windows} windows, got {len(predictions_per_window)}"
        )
    return predictions_per_window
