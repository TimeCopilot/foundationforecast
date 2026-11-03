from __future__ import annotations

import inspect
from collections.abc import Callable

import pandas as pd
import utilsforecast.processing as ufp

from .exog.covariates import (
    exog_strategy_disabled,
    normalize_exog_strategy,
    resolve_exog_columns_from_df,
    resolve_horizon_exog_df,
)
from .exog.futr_exog import futr_exog_unsupported_message
from .forecaster import Forecaster, maybe_infer_freq
from .utils import PanelData, process_panel_from_df


def _accepts_kwarg(fn: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return any(
        param.name == name or param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )


def _with_panel_kwargs(
    fn: Callable[..., object],
    known_kwargs: dict[str, object],
    panel: PanelData | None,
) -> dict[str, object]:
    call_kwargs = dict(known_kwargs)
    if panel is not None and _accepts_kwarg(fn, "panel"):
        call_kwargs["panel"] = panel
    return call_kwargs


def _optional_exog_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Omit exog keys unless the caller supplied horizon data or column names."""
    exog_keys = ("X_df", "futr_df", "futr_exog_list")
    if not any(kwargs.get(key) is not None for key in exog_keys):
        return {k: v for k, v in kwargs.items() if k not in exog_keys}
    filtered: dict[str, object] = {
        k: v for k, v in kwargs.items() if k not in exog_keys
    }
    for key in exog_keys:
        if kwargs.get(key) is not None:
            filtered[key] = kwargs[key]
    return filtered


def _model_uses_horizon_exog(
    model: Forecaster,
    *,
    attr: str,
    df: pd.DataFrame,
    kwargs: dict[str, object],
) -> bool:
    strategy = normalize_exog_strategy(getattr(model, "exog_strategy", "auto"))
    if exog_strategy_disabled(strategy):
        return False
    if attr == "forecast":
        horizon_df = resolve_horizon_exog_df(
            kwargs.get("X_df"),  # type: ignore[arg-type]
            kwargs.get("futr_df"),  # type: ignore[arg-type]
        )
        return horizon_df is not None
    if attr == "detect_anomalies" and kwargs.get("X_df") is not None:
        return True
    if attr in ("cross_validation", "detect_anomalies"):
        futr_exog_list = kwargs.get("futr_exog_list")
        cols = resolve_exog_columns_from_df(
            df,
            futr_exog_list if isinstance(futr_exog_list, list) else None,
        )
        return bool(cols)
    return False


def _validate_models_horizon_exog(
    models: list[Forecaster],
    *,
    attr: str,
    df: pd.DataFrame,
    kwargs: dict[str, object],
) -> None:
    for model in models:
        if not _model_uses_horizon_exog(model, attr=attr, df=df, kwargs=kwargs):
            continue
        if not model.supports_native_futr_exog():
            raise ValueError(futr_exog_unsupported_message(model))


class MultiModelForecasterMixin:
    """Orchestrate multiple Forecaster instances through a single interface."""

    models: list[Forecaster]
    fallback_model: Forecaster | None
    clean_cache: bool

    def _validate_unique_aliases(self, models: list[Forecaster]) -> None:
        aliases = [model.alias for model in models]
        duplicates = {a for a in aliases if aliases.count(a) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate model aliases found: {sorted(duplicates)}. "
                "Each model must have a unique alias."
            )

    def _clean_model_cache(self) -> None:
        import gc

        models_to_clear = list(self.models)
        if self.fallback_model is not None:
            models_to_clear.append(self.fallback_model)
        for model in models_to_clear:
            clear_cache = getattr(model, "clear_model_cache", None)
            if clear_cache is not None:
                clear_cache()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _call_models(
        self,
        attr: str,
        merge_on: list[str],
        df: pd.DataFrame,
        h: int | None,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        panel: PanelData | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        Forecaster.validate_input(df, h)
        freq = maybe_infer_freq(df, freq)
        if panel is None and attr == "forecast":
            panel = process_panel_from_df(df)
        _validate_models_horizon_exog(
            self.models,
            attr=attr,
            df=df,
            kwargs=kwargs,
        )
        res_df: pd.DataFrame | None = None
        for model in self.models:
            known_kwargs = {
                "df": df,
                "h": h,
                "freq": freq,
                "level": level,
            }
            if attr != "detect_anomalies":
                known_kwargs["quantiles"] = quantiles
            fn = getattr(model, attr)
            exog_kwargs = _optional_exog_kwargs(kwargs)
            call_kwargs = _with_panel_kwargs(fn, known_kwargs, panel)
            try:
                res_df_model = fn(**call_kwargs, **exog_kwargs)
            except (ValueError, RuntimeError) as e:
                if self.fallback_model is None:
                    raise e
                fn = getattr(self.fallback_model, attr)
                fallback_kwargs = _with_panel_kwargs(fn, known_kwargs, panel)
                res_df_model = fn(**fallback_kwargs, **exog_kwargs)
                res_df_model = res_df_model.rename(
                    columns={
                        col: (
                            col.replace(self.fallback_model.alias, model.alias)
                            if col.startswith(self.fallback_model.alias)
                            else col
                        )
                        for col in res_df_model.columns
                    }
                )
            if res_df is None:
                res_df = res_df_model
            else:
                if "y" in res_df_model:
                    res_df_model = res_df_model.drop(columns=["y"])
                res_df = ufp.join(res_df, res_df_model, on=merge_on, how="left")
            if self.clean_cache:
                self._clean_model_cache()
        if res_df is None:
            raise ValueError("At least one model is required.")
        return res_df

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        panel: PanelData | None = None,
        X_df: pd.DataFrame | None = None,
        *,
        futr_df: pd.DataFrame | None = None,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """Forecast the next `h` steps of every series with every model.

        Args:
            df: Historical data with columns `unique_id`, `ds`, and `y`, plus any
                exogenous columns.
            h: Forecast horizon, in steps of `freq`.
            freq: Pandas frequency of `ds`. Inferred from `df` when omitted.
            level: Prediction-interval levels in percent (e.g. `[80, 95]`).
                Cannot be combined with `quantiles`.
            quantiles: Quantiles in `(0, 1)`. Cannot be combined with `level`.
            panel: Precomputed panel data; internal, leave as `None`.
            X_df: Known-future exogenous values for the next `h` steps, with
                `unique_id`, `ds`, and the exogenous columns (no `y`).
            futr_df: Alias for `X_df`.
            futr_exog_list: Exogenous column names to use. Inferred from `X_df`
                when omitted.

        Returns:
            One row per series and future timestamp, with `unique_id`, `ds`, and
                one point-forecast column per model alias, plus that model's
                `-lo-`/`-hi-` or `-q-` columns.
        """
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
            panel=panel,
            X_df=X_df,
            futr_df=futr_df,
            futr_exog_list=futr_exog_list,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        *,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """Backtest every model over the same rolling windows.

        Args:
            df: Historical data with columns `unique_id`, `ds`, and `y`. To use
                exogenous variables, include them (and `y`) through the end of
                every window.
            h: Forecast horizon for each window.
            freq: Pandas frequency of `ds`. Inferred from `df` when omitted.
            n_windows: Number of windows.
            step_size: Steps between consecutive cutoffs. Defaults to `h`.
            level: Prediction-interval levels in percent. Cannot be combined
                with `quantiles`.
            quantiles: Quantiles in `(0, 1)`. Cannot be combined with `level`.
            futr_exog_list: Exogenous column names to use. Inferred from the
                non-target columns of `df` when omitted.

        Returns:
            One row per series, window, and timestamp, with `unique_id`, `ds`,
                `cutoff`, `y`, and one forecast column per model alias.
        """
        return self._call_models(
            "cross_validation",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
            n_windows=n_windows,
            step_size=step_size,
            futr_exog_list=futr_exog_list,
        )

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
        X_df: pd.DataFrame | None = None,
        *,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """Flag anomalies with every model.

        Each model backtests the history and marks values outside its `level`
        prediction interval. Exogenous variables are passed to every model;
        set `exog_strategy=False` on models without native exog support so
        they ignore them.

        Args:
            df: Historical data with columns `unique_id`, `ds`, and `y`. As in
                `cross_validation`, any exogenous columns in `df` are used as
                known-future covariates for each window.
            h: Forecast horizon for each window. Defaults to the seasonality
                implied by `freq`.
            freq: Pandas frequency of `ds`. Inferred from `df` when omitted.
            n_windows: Number of windows. Defaults to as many as the shortest
                series allows.
            level: Interval level in percent used as the anomaly threshold.
            X_df: Exogenous values with `unique_id`, `ds`, and the exogenous
                columns (no `y`), kept apart from `df`. Must have a row for
                every `(unique_id, ds)` in `df`.
            futr_exog_list: Exogenous column names to use. Inferred from `X_df`
                (or from the non-target columns of `df`) when omitted.

        Returns:
            One row per series and checked timestamp, with `unique_id`, `ds`,
                `cutoff`, `y`, and per model alias the forecast,
                `-lo-{level}`/`-hi-{level}` bounds, and a boolean `-anomaly` column.
        """
        return self._call_models(
            "detect_anomalies",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            level=level,  # type: ignore[arg-type]
            quantiles=None,
            n_windows=n_windows,
            X_df=X_df,
            futr_exog_list=futr_exog_list,
        )
