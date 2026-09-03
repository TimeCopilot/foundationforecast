from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from tafsut import TafsutModel
from tafsut import forecast as tafsut_forecast
from tqdm import tqdm

from ..core.forecaster import Forecaster, QuantileConverter, _DataProcessor
from ..core.utils import TimeSeriesDataset


class Tafsut(Forecaster, _DataProcessor):
    """
    Tafsut is a zero-shot probabilistic univariate time series foundation model.
    It uses a patch-based transformer encoder to produce nine forecast quantiles
    per step from up to 32,768 historical observations. See the
    [official repo](https://github.com/Tafsut-FM/tafsut) and
    [model card](https://huggingface.co/Tafsut-FM/tafsut-univariate-base) for
    more details.
    """

    def __init__(
        self,
        repo_id: str = "Tafsut-FM/tafsut-univariate-base",
        context_length: int = 32_768,
        batch_size: int = 64,
        alias: str = "Tafsut",
    ):
        """
        Initialize Tafsut time series foundation model.

        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Tafsut model from. Defaults to
                "Tafsut-FM/tafsut-univariate-base".
            context_length (int, optional): Maximum context length (input window
                size) for the model. Controls how much history is used for each
                forecast. Defaults to 32,768. Inputs longer than this are
                truncated to the most recent observations.
            batch_size (int, optional): Batch size for inference. Defaults to 64.
                Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "Tafsut".

        Notes:
            **Resources:**

            - GitHub: [Tafsut-FM/tafsut](https://github.com/Tafsut-FM/tafsut)
            - HuggingFace: [Tafsut-FM/tafsut-univariate-base](
                https://huggingface.co/Tafsut-FM/tafsut-univariate-base
              )

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - Tafsut outputs nine fixed quantiles (0.1 through 0.9).
            - Missing values in the context are handled natively via NaN.
            - Univariate only; no covariates or cross-series structure.
        """
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    @contextmanager
    def _get_model(self) -> TafsutModel:
        model = TafsutModel.from_pretrained(self.repo_id, device=self.device)
        try:
            model.eval()
            yield model
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _predict_batch(
        self,
        model: TafsutModel,
        batch: list[torch.Tensor],
        h: int,
        quantiles: list[float] | None,
        supported_quantiles: list[float],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        context = self._prepare_and_validate_context(batch)
        if context.shape[1] > self.context_length:
            context = context[..., -self.context_length :]
        context_np = context.detach().cpu().to(torch.float32).numpy()
        fcst = tafsut_forecast(model, context_np, horizon=h)
        if isinstance(fcst, torch.Tensor):
            fcst_np = fcst.detach().cpu().numpy()
        else:
            fcst_np = np.asarray(fcst)
        fcst_np = np.sort(fcst_np, axis=-1)
        fcst_mean_np = fcst_np[..., supported_quantiles.index(0.5)]
        fcst_quantiles_np = fcst_np if quantiles is not None else None
        return fcst_mean_np, fcst_quantiles_np

    def _predict(
        self,
        model: TafsutModel,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
        supported_quantiles: list[float],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        fcsts = [
            self._predict_batch(
                model,
                batch,
                h,
                quantiles,
                supported_quantiles,
            )
            for batch in tqdm(dataset)
        ]
        fcsts_mean_tp, fcsts_quantiles_tp = zip(*fcsts, strict=False)
        fcsts_mean_np = fcsts_mean_tp[0]
        if fcsts_mean_tp[0].shape != tuple():
            fcsts_mean_np = np.concatenate(fcsts_mean_tp)
        if quantiles is not None:
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles_tp)
        else:
            fcsts_quantiles_np = None
        return fcsts_mean_np, fcsts_quantiles_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
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
                = 100 × quantile value.

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
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            supported_quantiles = list(model.cfg.quantiles)
            if qc.quantiles is not None and not np.allclose(
                qc.quantiles,
                supported_quantiles,
            ):
                raise ValueError(
                    "Tafsut only supports the default quantiles, "
                    f"supported quantiles are {supported_quantiles}, "
                    "please use the default quantiles or default level, "
                )
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
                supported_quantiles=supported_quantiles,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None and fcsts_quantiles_np is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
