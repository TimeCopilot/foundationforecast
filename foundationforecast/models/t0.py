import json
import sys
from contextlib import contextmanager

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    raise ImportError("T0 requires Python >= 3.11 and < 3.14")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from t0 import T0Forecaster
from tqdm import tqdm

from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.quantiles import (
    T0_ALPHA_QUANTILE_RANGE,
    T0_BETA_QUANTILE_RANGE,
    backend_quantile_levels,
    select_clipped_quantile_values,
)
from ..core.utils import PanelData


class T0(ExogCapableForecaster):
    """
    T0 is an open-weights time series foundation model from
    [The Forecasting Company](https://theforecastingcompany.com/). It is a
    decoder-style patch transformer that alternates time and covariate
    attention layers, producing probabilistic multi-horizon quantile
    forecasts. It decodes up to 1,024 timesteps in a single forward pass and
    falls back on autoregressive rollout for longer horizons. T0 natively
    handles numerical covariates, both historical (known over the past) and
    future (known over the forecast horizon). See the
    [model card](https://huggingface.co/theforecastingcompany/t0-alpha)
    for more details.
    """

    def __init__(
        self,
        repo_id: str = "theforecastingcompany/t0-alpha",
        context_length: int = 4096,
        batch_size: int = 16,
        alias: str = "t0-alpha",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the T0 model from. Defaults to "theforecastingcompany/t0-alpha".
                See the full list of models at
                [Hugging Face](https://huggingface.co/theforecastingcompany).
            context_length (int, optional): Maximum context length (input window
                size) for the model. Series longer than this are truncated to the
                most recent `context_length` observations. Defaults to 4096.
            batch_size (int, optional): Batch size to use for inference. Defaults
                to 16. Adjust based on available memory.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "t0-alpha".
            reuse_loaded_model (bool, optional): When True (default), reuse
                loaded checkpoint weights across repeated ``forecast()`` calls
                via the process-wide LRU cache. Set to False to load and
                release weights on every call.

        Notes:
            **Requirements:**

            - T0 requires Python 3.11 to 3.13 and ``tfc-t0>=0.5.0`` (required
              for ``t0-beta``; also supports ``t0-alpha``).

            **Available models:**

            | Model ID                                                                                          | Parameters |
            | ------------------------------------------------------------------------------------------------- | ---------- |
            | [`theforecastingcompany/t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha)         | ~102M      |
            | [`theforecastingcompany/t0-beta`](https://huggingface.co/theforecastingcompany/t0-beta)           | ~256M      |

            **Resources:**

            - HuggingFace: [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha)
            - Platform: [Retrocast](https://app.retrocast.com/)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - ``t0-alpha`` predicts 5 quantile knots (0.1, 0.25, 0.5, 0.75, 0.9);
              ``t0-beta`` predicts 21 native levels (0.01–0.99). Requested
              quantiles are interpolated; the median (0.5) is the point forecast.
            - NaN values in the context are treated as missing observations.
            - Known-future exogenous variables are passed via ``X_df`` at forecast
              time (``exog_strategy="auto"``).
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
        )
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> T0Forecaster:
        # huggingface_hub may not inject config.json into model kwargs when the
        # checkpoint repo is gated; pass the config explicitly.
        config_path = hf_hub_download(self.repo_id, CONFIG_NAME)
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return (
            T0Forecaster.from_pretrained(self.repo_id, **config).to(self.device).eval()
        )

    @contextmanager
    def _get_model(self) -> T0Forecaster:
        with self._cached_model(self._load_model) as model:
            yield model

    def _quantile_range(self) -> tuple[float, float]:
        if "beta" in self.repo_id:
            return T0_BETA_QUANTILE_RANGE
        return T0_ALPHA_QUANTILE_RANGE

    def _batch_context_width(self, batch: list[torch.Tensor]) -> int:
        return min(max(len(ts) for ts in batch), self.context_length)

    def _to_context(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Left-pad a ragged batch with NaN (treated as missing by T0)."""
        max_len = self._batch_context_width(batch)
        context = torch.full(
            (len(batch), max_len),
            float("nan"),
            dtype=torch.float32,
        )
        for idx, ts in enumerate(batch):
            ts = ts[-max_len:]
            context[idx, -len(ts) :] = ts.to(dtype=torch.float32)
        return context

    def _to_future_covariates(
        self,
        batch: list[torch.Tensor],
        uids: list | np.ndarray,
        df: pd.DataFrame,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
        h: int,
    ) -> np.ndarray:
        """Build ``[B, F, T + h]`` arrays for T0 ``future_covariates``."""
        max_len = self._batch_context_width(batch)
        n_features = len(futr_exog_list)
        covariates = np.full(
            (len(batch), n_features, max_len + h),
            np.nan,
            dtype=np.float32,
        )
        for idx, uid in enumerate(uids):
            hist = df.loc[df["unique_id"] == uid, futr_exog_list].to_numpy(
                dtype=np.float32
            )
            if hist.shape[0] > max_len:
                hist = hist[-max_len:]
            hor = horizon_df.loc[
                horizon_df["unique_id"] == uid, futr_exog_list
            ].to_numpy(dtype=np.float32)
            if hor.shape[0] != h:
                raise ValueError("Horizon exog length mismatch.")
            n_hist = hist.shape[0]
            covariates[idx, :, max_len - n_hist : max_len] = hist.T
            covariates[idx, :, max_len:] = hor.T
        return covariates

    def supports_native_futr_exog(self) -> bool:
        return True

    def _forecast_native_futr_exog(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        panel: PanelData | None,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
    ) -> pd.DataFrame | None:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, panel=panel
        )
        fcst_df = horizon_df[["unique_id", "ds"]].copy()
        q_min, q_max = self._quantile_range()
        if qc.quantiles is not None:
            pred_quantiles = backend_quantile_levels(
                qc.quantiles,
                q_min=q_min,
                q_max=q_max,
                include_median=True,
            )
        else:
            pred_quantiles = [0.5]
        median_idx = pred_quantiles.index(float(np.clip(0.5, q_min, q_max)))
        fcsts: list[np.ndarray] = []
        uids = np.asarray(dataset.uids)
        with self._get_model() as model:
            for batch in tqdm(dataset):
                batch_len = len(batch)
                start = (dataset.current_batch - 1) * dataset.batch_size
                batch_uids = uids[start : start + batch_len]
                future_cov = self._to_future_covariates(
                    batch,
                    batch_uids,
                    df,
                    horizon_df,
                    futr_exog_list,
                    h,
                )
                out = model.predict(
                    self._to_context(batch),
                    horizon=h,
                    quantile_levels=pred_quantiles,
                    future_covariates=future_cov,
                )
                fcsts.append(out.quantiles.cpu().numpy())
        fcsts_np = np.concatenate(fcsts, axis=0)
        fcst_df[self.alias] = fcsts_np[..., median_idx].reshape(-1, 1)
        if qc.quantiles is not None:
            fcsts_quantiles_np = select_clipped_quantile_values(
                pred_quantiles,
                fcsts_np,
                qc.quantiles,
                q_min=q_min,
                q_max=q_max,
                axis=-1,
            )
            fcst_df = self._assign_quantile_forecasts(
                fcst_df,
                self.alias,
                qc.quantiles,
                fcsts_quantiles_np,
            )
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df

    def _forecast_univariate(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        panel: PanelData | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

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
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value. Quantiles the model wasn't trained on
                are linearly interpolated across its fixed knots.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, panel=panel
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        q_min, q_max = self._quantile_range()
        if qc.quantiles is not None:
            pred_quantiles = backend_quantile_levels(
                qc.quantiles,
                q_min=q_min,
                q_max=q_max,
                include_median=True,
            )
        else:
            pred_quantiles = [0.5]
        median_idx = pred_quantiles.index(float(np.clip(0.5, q_min, q_max)))
        fcsts: list[np.ndarray] = []
        with self._get_model() as model:
            for batch in tqdm(dataset):
                out = model.predict(
                    self._to_context(batch),
                    horizon=h,
                    quantile_levels=pred_quantiles,
                )
                # shape: (batch, h, n_quantiles)
                fcsts.append(out.quantiles.cpu().numpy())
        fcsts_np = np.concatenate(fcsts, axis=0)
        fcst_df[self.alias] = fcsts_np[..., median_idx].reshape(-1, 1)
        if qc.quantiles is not None:
            fcsts_quantiles_np = select_clipped_quantile_values(
                pred_quantiles,
                fcsts_np,
                qc.quantiles,
                q_min=q_min,
                q_max=q_max,
                axis=-1,
            )
            fcst_df = self._assign_quantile_forecasts(
                fcst_df,
                self.alias,
                qc.quantiles,
                fcsts_quantiles_np,
            )
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
