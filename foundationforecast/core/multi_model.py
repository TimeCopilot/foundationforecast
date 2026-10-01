from __future__ import annotations

import inspect
from collections.abc import Callable

import pandas as pd
import utilsforecast.processing as ufp

from .forecaster import Forecaster, maybe_infer_freq
from .panel_columns import resolve_panel_columns
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
        df: pd.DataFrame,
        h: int | None,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        panel: PanelData | None = None,
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        cols = resolve_panel_columns(self, id_col, time_col, target_col)
        Forecaster.validate_input(df, h, cols)
        df_work = cols.to_canonical(df)
        freq = maybe_infer_freq(df_work, freq)
        if panel is None and attr == "forecast":
            panel = process_panel_from_df(df_work)
        merge_on = cols.merge_keys() if attr == "forecast" else cols.cv_merge_keys()
        if "X_df" in kwargs and kwargs["X_df"] is not None:
            kwargs = dict(kwargs)
            kwargs["X_df"] = cols.to_canonical(kwargs["X_df"])
        if "futr_df" in kwargs and kwargs["futr_df"] is not None:
            kwargs = dict(kwargs)
            kwargs["futr_df"] = cols.to_canonical(kwargs["futr_df"])
        res_df: pd.DataFrame | None = None
        for model in self.models:
            known_kwargs = {
                "df": df,
                "h": h,
                "freq": freq,
                "level": level,
                "id_col": cols.id_col,
                "time_col": cols.time_col,
                "target_col": cols.target_col,
            }
            if attr != "detect_anomalies":
                known_kwargs["quantiles"] = quantiles
            fn = getattr(model, attr)
            call_kwargs = _with_panel_kwargs(fn, known_kwargs, panel)
            try:
                res_df_model = fn(**call_kwargs, **kwargs)
            except (ValueError, RuntimeError) as e:
                if self.fallback_model is None:
                    raise e
                fn = getattr(self.fallback_model, attr)
                fallback_kwargs = _with_panel_kwargs(fn, known_kwargs, panel)
                res_df_model = fn(**fallback_kwargs, **kwargs)
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
                if cols.target_col in res_df_model:
                    res_df_model = res_df_model.drop(columns=[cols.target_col])
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
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "forecast",
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
            panel=panel,
            X_df=X_df,
            futr_df=futr_df,
            futr_exog_list=futr_exog_list,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
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
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "cross_validation",
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
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "detect_anomalies",
            df=df,
            h=h,
            freq=freq,
            level=level,  # type: ignore[arg-type]
            quantiles=None,
            n_windows=n_windows,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
