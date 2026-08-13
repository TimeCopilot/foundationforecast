from __future__ import annotations

import pandas as pd

from .core.forecaster import Forecaster


class FoundationForecast(Forecaster):
    """Unified forecaster for multiple foundation time series models.

    This class runs multiple pretrained foundation
    models through a single interface and merges their forecasts.
    """

    alias = "FoundationForecast"

    def __init__(
        self,
        models: list[Forecaster],
        fallback_model: Forecaster | None = None,
        clean_cache: bool = False,
    ):
        self._validate_unique_aliases(models)
        self.models = models
        self.fallback_model = fallback_model
        self.clean_cache = clean_cache

    def _validate_unique_aliases(self, models: list[Forecaster]) -> None:
        aliases = [model.alias for model in models]
        duplicates = {a for a in aliases if aliases.count(a) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate model aliases found: {sorted(duplicates)}. "
                "Each model must have a unique alias."
            )

    @staticmethod
    def _clean_model_cache() -> None:
        import gc

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
        **kwargs,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
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
            try:
                res_df_model = fn(**known_kwargs, **kwargs)
            except (ValueError, RuntimeError) as e:
                if self.fallback_model is None:
                    raise e
                fn = getattr(self.fallback_model, attr)
                res_df_model = fn(**known_kwargs, **kwargs)
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
                res_df = res_df.merge(res_df_model, on=merge_on, how="left")
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
    ) -> pd.DataFrame:
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
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
