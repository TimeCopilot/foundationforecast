from __future__ import annotations

import inspect
from collections.abc import Callable

import pandas as pd
import utilsforecast.processing as ufp

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
        X_df: pd.DataFrame | None = None,
        *,
        futr_df: pd.DataFrame | None = None,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
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
            X_df=X_df,
            futr_df=futr_df,
            futr_exog_list=futr_exog_list,
        )

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
    ) -> pd.DataFrame:
        return self._call_models(
            "detect_anomalies",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            level=level,  # type: ignore[arg-type]
            quantiles=None,
            n_windows=n_windows,
        )
