from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.quantiles import resolve_quantile_values
from ..core.utils import PanelData, TimeSeriesDataset

TABBY_QUANTILES = [round(i / 100, 2) for i in range(1, 100)]


class Tabby(ExogCapableForecaster):
    """
    Tabby is a zero-shot probabilistic univariate time series foundation model
    from [Huawei Paris Noah's Ark Lab](https://huggingface.co/paris-noah).
    It is a prompted encoder-only patch Transformer (~147M) that emits 99
    quantile forecasts (0.01 through 0.99) from up to 8,192 historical
    observations. See the [paper](https://arxiv.org/abs/2609.13956) and
    [model card](https://huggingface.co/paris-noah/Tabby) for more details.
    """

    def __init__(
        self,
        repo_id: str = "paris-noah/Tabby",
        context_length: int = 8096,
        batch_size: int = 64,
        alias: str = "Tabby",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        precision: Literal["bf16", "fp32"] = "bf16",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Tabby model from. Defaults to ``paris-noah/Tabby``.
                The released checkpoint is a self-contained prompted model
                (``AutoModel`` + ``trust_remote_code``).
            context_length (int, optional): Maximum history length passed to the
                model. Defaults to 8,096 (the published GIFT-Eval input cap).
                The backbone window is 8,192; for horizon ``H`` the model uses at
                most ``8192 - max(H, 128)`` observed steps. Longer inputs are
                truncated to the most recent observations.
            batch_size (int, optional): Batch size for inference. Defaults to 64.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to ``Tabby``.
            reuse_loaded_model (bool, optional): When True (default), reuse
                loaded checkpoint weights across repeated ``forecast()`` calls
                via the process-wide LRU cache. Set to False to load and
                release weights on every call.
            precision (str, optional): ``"bf16"`` (default) runs the forward
                pass under bfloat16 autocast on CUDA, as in the published
                GIFT-Eval evaluation; ``"fp32"`` disables autocast. On CPU the
                model always runs in float32.

        Notes:
            **Academic Reference:**

            - Paper: [Tabby: An Open Pretraining Recipe for Time Series Foundation Models](https://arxiv.org/abs/2609.13956)

            **Resources:**

            - HuggingFace: [paris-noah/Tabby](https://huggingface.co/paris-noah/Tabby)
            - Code: [huawei-noah/trustworthyAI](https://github.com/huawei-noah/trustworthyAI)

            **Technical Details:**

            - Requires ``trust_remote_code=True`` (custom ``TabbyForForecasting``).
            - 99 native quantile knots (0.01–0.99); the median (0.5) is the
              point forecast. Requested ``level`` / ``quantiles`` are linearly
              interpolated with edge clamping.
            - Missing values in the context are handled natively via NaN;
              left padding of shorter series in a batch is flagged as padding
              (``past_is_pad``) rather than missing data.
            - Univariate only; no covariates.
            - Weights are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
        )
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> Any:
        return (
            AutoModel.from_pretrained(self.repo_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    @contextmanager
    def _get_model(self) -> Iterator[Any]:
        with self._cached_model(self._load_model) as model:
            yield model

    def _native_quantiles(self, model: Any) -> list[float]:
        levels = getattr(model, "quantile_levels", None)
        if levels:
            return [float(q) for q in levels]
        return list(TABBY_QUANTILES)

    def _to_context(
        self, batch: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a ragged batch, returning ``(context, past_is_pad)``."""
        series = [ts[-self.context_length :].to(torch.float32) for ts in batch]
        width = max(len(ts) for ts in series)
        context = torch.full((len(series), width), float("nan"))
        past_is_pad = torch.zeros((len(series), width), dtype=torch.bool)
        for idx, ts in enumerate(series):
            context[idx, width - len(ts) :] = ts
            past_is_pad[idx, : width - len(ts)] = True
        return context, past_is_pad

    def _predict_batch(
        self,
        model: Any,
        batch: list[torch.Tensor],
        h: int,
        quantiles: list[float] | None,
        supported_quantiles: list[float],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        context, past_is_pad = self._to_context(batch)
        context = context.to(self.device)
        use_bf16 = self.precision == "bf16" and self.device.type == "cuda"
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16),
        ):
            out = model(
                context=context,
                context_mask=torch.isfinite(context).float(),
                past_is_pad=past_is_pad.to(self.device),
                prediction_length=h,
            )["quantile_preds"]
        # (batch_size, 99, h) -> (batch_size, h, 99)
        fcst_np = np.moveaxis(out.float().cpu().numpy(), 1, -1)
        median_idx = supported_quantiles.index(0.5)
        fcst_mean_np = fcst_np[..., median_idx]
        fcst_quantiles_np = fcst_np if quantiles is not None else None
        return fcst_mean_np, fcst_quantiles_np

    def _predict(
        self,
        model: Any,
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
                = 100 × quantile value.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.

        Examples:
            ```python
            import pandas as pd
            from foundationforecast.models import Tabby

            df = pd.read_csv(
                "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
                parse_dates=["ds"],
            )
            model = Tabby()
            fcst = model.forecast(df, h=12, freq="MS", level=[90])
            assert "Tabby" in fcst.columns
            ```
        """
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = self._make_timeseries_dataset(
            df,
            batch_size=self.batch_size,
            dtype=torch.float32,
            panel=panel,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            supported_quantiles = self._native_quantiles(model)
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
                quantiles=supported_quantiles if qc.quantiles is not None else None,
                supported_quantiles=supported_quantiles,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None and fcsts_quantiles_np is not None:
            fcsts_quantiles_np = resolve_quantile_values(
                supported_quantiles,
                fcsts_quantiles_np,
                qc.quantiles,
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
