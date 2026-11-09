from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from kumo_forecast import KumoForecastModel, load_kumo_forecast
from tqdm import tqdm

from ..core.conformal import ConformalIntervals
from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, _DataProcessor
from ..core.utils import PanelData, TimeSeriesDataset


class Kumo(ExogCapableForecaster, _DataProcessor):
    """
    Kumo-Forecast is NVIDIA's zero-shot univariate forecaster built on a
    MOMENT-style T5 encoder with a linear forecasting head. It reads up to 512
    historical observations and predicts 72 steps at once; longer horizons roll
    the forecast forward autoregressively. See the
    [official repo](https://github.com/NVIDIA/Kumo-TS) and
    [model card](https://huggingface.co/nvidia/Kumo-Forecast) for more details.

    Kumo-Forecast produces **point forecasts only**. Requesting `level` or
    `quantiles` raises a `ValueError` unless the model is created with
    `prediction_intervals=ConformalIntervals(...)`, which calibrates intervals
    from cross-validation errors of the point forecast.
    """

    def __init__(
        self,
        repo_id: str = "nvidia/Kumo-Forecast",
        checkpoint: str = "run8_best_model_cr.pt",
        context_length: int = 512,
        batch_size: int = 16,
        alias: str = "Kumo",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Initialize the Kumo-Forecast time series foundation model.

        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local
                directory holding the checkpoint. Defaults to
                "nvidia/Kumo-Forecast".
            checkpoint (str, optional): Checkpoint file inside `repo_id`.
                Defaults to "run8_best_model_cr.pt" (with cross-channel
                attention). The other published checkpoint is
                "moment_head_512_6hr.pt" (without it).
            context_length (int, optional): Maximum number of recent
                observations used per series. The published checkpoint reads
                at most 512; shorter series are left-padded and masked.
                Defaults to 512.
            batch_size (int, optional): Number of series per forward pass.
                Defaults to 16.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "Kumo".
            reuse_loaded_model (bool, optional): When True (default), reuse
                loaded checkpoint weights across repeated ``forecast()`` calls
                via the process-wide LRU cache. Set to False to load and
                release weights on every call.
            exog_strategy (ExogStrategyConfig, optional): How to handle
                known-future exogenous variables. Kumo has no native support.
                Defaults to "auto".
            prediction_intervals (ConformalIntervals | None, optional): Required
                to request `level` or `quantiles`. ``forecast()`` then calibrates
                intervals on cross-validation errors of the point forecast. See
                the conformal prediction docs. Defaults to None.

        Notes:
            **Resources:**

            - GitHub: [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS)
            - HuggingFace: [nvidia/Kumo-Forecast](
                https://huggingface.co/nvidia/Kumo-Forecast
              )

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU) and has about 340M parameters.
            - Point forecasts only; no native quantiles or intervals. Use
              `prediction_intervals` for conformal intervals.
            - Missing values in the context are handled natively via a mask.
            - Univariate: each series is forecast independently, without
              covariates or cross-series structure.
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
            prediction_intervals=prediction_intervals,
        )
        if context_length < 1:
            raise ValueError("`context_length` must be a positive integer.")
        self.repo_id = repo_id
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    def _load_model(self) -> KumoForecastModel:
        return load_kumo_forecast(
            self.repo_id,
            checkpoint=self.checkpoint,
            device=self.device,
            dtype=self.dtype,
        )

    def supports_native_quantiles(self) -> bool:
        return False

    @contextmanager
    def _get_model(self):
        key = f"{self._model_cache_prefix()}:{self.checkpoint}:{self.device}"
        with self._cached_model(self._load_model, cache_key=key) as model:
            yield model

    def _predict(
        self,
        model: KumoForecastModel,
        dataset: TimeSeriesDataset,
        h: int,
    ) -> np.ndarray:
        fcsts = []
        for batch in tqdm(dataset):
            context = self._prepare_and_validate_context(batch)
            context = context[..., -self.context_length :]
            context_np = context.detach().cpu().to(torch.float32).numpy()
            fcsts.append(model.forecast(context_np, h))
        return np.concatenate(fcsts)

    def _forecast_univariate(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        panel: PanelData | None = None,
    ) -> pd.DataFrame:
        """Generate point forecasts for time series data using the model.

        The input DataFrame can contain one or multiple time series in stacked
        (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Not supported natively. Raises a `ValueError`; create the model
                with `prediction_intervals` to get conformal intervals.
            quantiles (list[float], optional):
                Not supported natively. Raises a `ValueError`; create the model
                with `prediction_intervals` to get conformal quantiles.

        Returns:
            pd.DataFrame:
                DataFrame with `unique_id`, `ds` and the point forecast for
                each timestamp and series, in the model alias column.
        """
        if level is not None or quantiles is not None:
            raise ValueError(
                f"{self.alias} produces point forecasts only, so `level` and "
                "`quantiles` need conformal intervals: create the model with "
                "`prediction_intervals=ConformalIntervals(...)`."
            )
        freq = self._maybe_infer_freq(df, freq)
        dataset = self._make_timeseries_dataset(
            df,
            batch_size=self.batch_size,
            dtype=self.dtype,
            panel=panel,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_np = self._predict(model, dataset, h)
        fcst_df[self.alias] = fcsts_np.reshape(-1, 1)
        return fcst_df
