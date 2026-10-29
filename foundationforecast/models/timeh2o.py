import sys
from contextlib import contextmanager

if sys.version_info < (3, 12):
    raise ImportError("TimeH2O requires Python >= 3.12")

import numpy as np
import pandas as pd
import torch
from h2ofm import TimeH2O as H2OTimeH2O
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from tqdm import tqdm

from ..core.conformal import ConformalIntervals
from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.utils import PanelData, TimeSeriesDataset


class TimeH2O(ExogCapableForecaster):
    """
    TimeH2O is a time series foundation model from
    [H2O.ai](https://h2o.ai/). It produces probabilistic multi-horizon
    quantile forecasts from a patch transformer with cross-variate attention,
    and natively handles known-future covariates (numeric and categorical) on
    the full checkpoint. See the
    [model card](https://huggingface.co/h2oai/timeh2o-v1) for more details.
    """

    def __init__(
        self,
        repo_id: str = "h2oai/timeh2o-v1",
        context_length: int | None = None,
        batch_size: int = 256,
        alias: str = "TimeH2O",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        prediction_intervals: ConformalIntervals | None = None,
        clip_nonnegative: bool = False,
        snap_integer: bool = False,
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the TimeH2O model from. Defaults to "h2oai/timeh2o-v1".
            context_length (int, optional): Maximum context length (input window
                size). Series longer than this are truncated to the most recent
                `context_length` observations. Defaults to None, which uses the
                checkpoint's own context window.
            batch_size (int, optional): Number of series per inference batch;
                also passed to ``h2ofm`` as its row budget (target plus
                covariate rows). Defaults to 256.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "TimeH2O".
            prediction_intervals (ConformalIntervals | None, optional): When set,
                ``forecast()`` replaces native quantiles with conformal intervals
                calibrated on cross-validation errors of the point forecast. See
                the conformal prediction docs. Defaults to None.
            reuse_loaded_model (bool, optional): When True (default), reuse
                loaded checkpoint weights across repeated ``forecast()`` calls
                via the process-wide LRU cache. Set to False to load and
                release weights on every call.
            exog_strategy (ExogStrategyConfig, optional): How exogenous
                variables in ``X_df`` are handled. Defaults to "auto", which
                passes known-future covariates to ``h2oai/timeh2o-v1``
                natively. ``h2oai/timeh2o-v1-tiny`` has no covariate support
                and raises an error when ``X_df`` is passed.
            clip_nonnegative (bool, optional): Clamp forecasts at zero for
                series whose history is nonnegative. Defaults to False.
            snap_integer (bool, optional): Round forecasts to integers for
                series whose history is integer-valued. Defaults to False.

        Notes:
            **Requirements:**

            - TimeH2O requires Python >= 3.12 and ``h2ofm>=0.43.0``.
            - The checkpoints are gated on Hugging Face: accept the terms on the
              model page and authenticate (e.g. ``HF_TOKEN``).
            - The weights are released under a **non-commercial** license.

            **Available models:**

            | Model ID                                                                | Parameters | Future covariates |
            | ----------------------------------------------------------------------- | ---------- | ----------------- |
            | [`h2oai/timeh2o-v1`](https://huggingface.co/h2oai/timeh2o-v1)           | ~109M      | Yes               |
            | [`h2oai/timeh2o-v1-tiny`](https://huggingface.co/h2oai/timeh2o-v1-tiny) | ~5.8M      | No                |

            **Resources:**

            - HuggingFace: [h2oai/timeh2o-v1](https://huggingface.co/h2oai/timeh2o-v1)
            - Package: [h2ofm](https://pypi.org/project/h2ofm/)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - The trained quantile grid spans 0.1–0.9; ``h2ofm`` extrapolates
              requested levels outside it with Gaussian tails. The median
              (0.5) is the point forecast.
            - The published GIFT-Eval and fev-bench scores enable
              ``clip_nonnegative`` and ``snap_integer``.
            - NaN values in the context are treated as missing observations.
            - Non-numeric covariates (strings, booleans, categoricals) are
              passed as categorical rows and encoded by ``h2ofm``.
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
            prediction_intervals=prediction_intervals,
        )
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.clip_nonnegative = clip_nonnegative
        self.snap_integer = snap_integer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> H2OTimeH2O:
        return H2OTimeH2O.from_pretrained(
            self.repo_id,
            device=self.device,
            batch_size=self.batch_size,
        )

    @contextmanager
    def _get_model(self) -> H2OTimeH2O:
        with self._cached_model(self._load_model) as model:
            yield model

    def supports_native_futr_exog(self) -> bool:
        return "tiny" not in self.repo_id

    def _context_width(self, ts: torch.Tensor) -> int:
        if self.context_length is None:
            return len(ts)
        return min(len(ts), self.context_length)

    def _to_contexts(self, batch: list[torch.Tensor]) -> list[np.ndarray]:
        return [ts[-self._context_width(ts) :].numpy() for ts in batch]

    def _to_future_covariates(
        self,
        batch: list[torch.Tensor],
        uids: list | np.ndarray,
        df: pd.DataFrame,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
        h: int,
    ) -> list[list[np.ndarray]]:
        """Build one list of ``T + h`` covariate rows per series.

        Rows keep their dtype so ``h2ofm`` encodes non-numeric ones as
        categorical.
        """
        covariates = []
        for ts, uid in zip(batch, uids, strict=True):
            n_hist = self._context_width(ts)
            hist = df.loc[df["unique_id"] == uid, futr_exog_list].iloc[-n_hist:]
            hor = horizon_df.loc[horizon_df["unique_id"] == uid, futr_exog_list]
            if len(hor) != h:
                raise ValueError("Horizon exog length mismatch.")
            rows = []
            for col in futr_exog_list:
                row = np.concatenate([hist[col].to_numpy(), hor[col].to_numpy()])
                if is_numeric_dtype(row.dtype) and not is_bool_dtype(row.dtype):
                    row = row.astype(np.float32)
                rows.append(row)
            covariates.append(rows)
        return covariates

    def _predict(
        self,
        model: H2OTimeH2O,
        contexts: list[np.ndarray],
        h: int,
        quantile_levels: list[float],
        future_covariates: list[list[np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Return forecasts shaped ``(n_series, h, n_levels)``."""
        forecasts = model.forecast_many(
            contexts,
            horizon=h,
            quantile_levels=quantile_levels,
            future_covariates=future_covariates,
            context_length=self.context_length,
            clip_nonnegative=self.clip_nonnegative,
            snap_integer=self.snap_integer,
        )
        return np.stack([np.asarray(f.quantiles) for f in forecasts], axis=0)

    def _forecast(
        self,
        df: pd.DataFrame,
        h: int,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        dataset: TimeSeriesDataset,
        fcst_df: pd.DataFrame,
        horizon_df: pd.DataFrame | None = None,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None:
            pred_levels = sorted({float(q) for q in qc.quantiles} | {0.5})
        else:
            pred_levels = [0.5]
        uids = np.asarray(dataset.uids)
        fcsts: list[np.ndarray] = []
        with self._get_model() as model:
            for batch in tqdm(dataset):
                future_cov = None
                if horizon_df is not None and futr_exog_list:
                    start = (dataset.current_batch - 1) * dataset.batch_size
                    future_cov = self._to_future_covariates(
                        batch,
                        uids[start : start + len(batch)],
                        df,
                        horizon_df,
                        futr_exog_list,
                        h,
                    )
                fcsts.append(
                    self._predict(
                        model,
                        self._to_contexts(batch),
                        h,
                        pred_levels,
                        future_covariates=future_cov,
                    )
                )
        fcsts_np = np.concatenate(fcsts, axis=0)
        fcst_df[self.alias] = fcsts_np[..., pred_levels.index(0.5)].reshape(-1, 1)
        if qc.quantiles is not None:
            idx = [pred_levels.index(float(q)) for q in qc.quantiles]
            fcst_df = self._assign_quantile_forecasts(
                fcst_df,
                self.alias,
                qc.quantiles,
                fcsts_np[..., idx],
            )
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df

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
        dataset = self._make_timeseries_dataset(
            df[["unique_id", "ds", "y"]],
            batch_size=self.batch_size,
            dtype=torch.float32,
            panel=panel,
        )
        return self._forecast(
            df,
            h,
            level,
            quantiles,
            dataset,
            fcst_df=horizon_df[["unique_id", "ds"]].copy(),
            horizon_df=horizon_df,
            futr_exog_list=futr_exog_list,
        )

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
        """
        freq = self._maybe_infer_freq(df, freq)
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, dtype=torch.float32, panel=panel
        )
        return self._forecast(
            df,
            h,
            level,
            quantiles,
            dataset,
            fcst_df=dataset.make_future_dataframe(h=h, freq=freq),
        )
