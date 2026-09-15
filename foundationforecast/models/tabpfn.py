import sys
from contextlib import contextmanager

if sys.version_info >= (3, 13):
    raise ImportError("TabPFN requires Python < 3.13")

import numpy as np
import pandas as pd
import torch
from tabpfn_client import set_access_token
from tabpfn_time_series import (
    DEFAULT_QUANTILE_CONFIG,
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)

from ..core.forecaster import Forecaster, QuantileConverter

TABPFN_V2_MODEL = "tabpfn-v2-regressor-2noar4o2.ckpt"
TABPFN_V3_MODEL = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"


class TabPFN(Forecaster):
    """
    TabPFN is a zero-shot time series forecasting model that frames univariate
    forecasting as a tabular regression problem using TabPFN. It supports both
    point and probabilistic forecasts, and can incorporate exogenous variables via
    feature engineering. This class supports TabPFN-2 (default) and TabPFN-3
    checkpoints. See the
    [official repo](https://github.com/PriorLabs/tabpfn-time-series) for more details.
    """

    def __init__(
        self,
        features: list[FeatureGenerator] | None = None,
        context_length: int = 4096,
        mode: TabPFNMode | None = None,
        api_key: str | None = None,
        alias: str = "TabPFN",
        reuse_loaded_model: bool = True,
        model_path: str = TABPFN_V2_MODEL,
    ):
        """
        Args:
            model_path (str, optional): TabPFN checkpoint filename. Defaults to
                TabPFN-2 (`tabpfn-v2-regressor-2noar4o2.ckpt`). Pass
                `tabpfn-v3-regressor-v3_20260506_timeseries.ckpt` for TabPFN-3
                (TabPFN-TS-3). See
                [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3).
            features (list[FeatureGenerator], optional): List of TabPFN-TS feature
                generators to use for feature engineering. If None, uses
                `[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]`
                by default.
                See
                [TabPFN-TS features](https://github.com/PriorLabs/tabpfn-time-series/
                tree/main/tabpfn_time_series/features).
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 4096. TabPFN-3 upstream default is 32768;
                increase for long-history tasks.
            mode (TabPFNMode, optional): Inference mode for TabPFN. If None, uses LOCAL
                if a GPU is available, otherwise CLIENT (cloud inference via
                tabpfn-client). See
                [TabPFN-TS docs](https://github.com/PriorLabs/tabpfn-time-series)
                for available modes.
            api_key (str, optional): API key for tabpfn-client cloud inference. Required
                if using CLIENT mode and not already set in the environment.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "TabPFN".
            reuse_loaded_model (bool, optional): When True (default), reuse
                loaded checkpoint weights across repeated ``forecast()`` calls
                via the process-wide LRU cache. Set to False to load and
                release weights on every call.

        Notes:
            **Academic Reference:**

            - Paper: [From Tables to Time: How TabPFN-v2 Outperforms
            Specialized Time Series Forecasting Models](https://arxiv.org/abs/2501.02945)
            - TabPFN-3: [TabPFN-3 report](https://priorlabs.ai/reports/tabpfn-3)

            **Resources:**

            - GitHub: [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series)

            **License:**

            - TabPFN-2.6+ and TabPFN-3 weights use the TabPFN Non-Commercial license.
            - First LOCAL use requires accepting terms at
              [ux.priorlabs.ai](https://ux.priorlabs.ai) (`TABPFN_TOKEN`).
            - Production/commercial use requires a
              [Prior Labs commercial license or API](https://docs.priorlabs.ai/models).

            **Technical Details:**

            - TabPFN-3 is currently LOCAL-only; the cloud client does not ship v3 yet.
            - For LOCAL mode, a CUDA-capable GPU is recommended for best performance.
            - The model is only available for Python < 3.13.
        """
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        if features is None:
            features = [
                RunningIndexFeature(),
                CalendarFeature(),
                AutoSeasonalFeature(),
            ]
        self.model_path = model_path
        self.feature_transformer = FeatureTransformer(features)
        self.context_length = context_length
        if mode is None:
            mode = TabPFNMode.LOCAL if torch.cuda.is_available() else TabPFNMode.CLIENT
        if model_path == TABPFN_V3_MODEL and mode != TabPFNMode.LOCAL:
            raise ValueError("TabPFN-3 is LOCAL-only; pass mode=TabPFNMode.LOCAL.")
        if mode == TabPFNMode.CLIENT and api_key is not None:
            set_access_token(api_key)
        self.mode = mode
        self.alias = alias

    def _model_cache_prefix(self) -> str | None:
        return f"TabPFN:{self.mode}:{self.model_path}"

    def _load_model(self) -> TabPFNTimeSeriesPredictor:
        return TabPFNTimeSeriesPredictor(
            tabpfn_mode=self.mode,
            tabpfn_config={"model_path": self.model_path},
        )

    @contextmanager
    def _get_model(self) -> TabPFNTimeSeriesPredictor:
        with self._cached_model(self._load_model) as model:
            yield model

    def _forecast(
        self,
        model: TabPFNTimeSeriesPredictor,
        df: pd.DataFrame,
        h: int,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        """handles distinction between quantiles and no quantiles"""
        renamer = {
            "unique_id": "item_id",
            "ds": "timestamp",
            "y": "target",
        }
        tsdf = df.rename(columns=renamer)
        tsdf["item_id"] = tsdf["item_id"].astype(str)
        tsdf = TimeSeriesDataFrame(tsdf.set_index(["item_id", "timestamp"]))
        if self.context_length > 0:
            tsdf = tsdf.slice_by_timestep(-self.context_length, None)
        future_tsdf = generate_test_X(tsdf, h)
        tsdf, future_tsdf = self.feature_transformer.transform(tsdf, future_tsdf)
        if quantiles is None:
            fcst_df = model.predict(tsdf, future_tsdf)
        else:
            fcst_df = model.predict(tsdf, future_tsdf, quantiles=quantiles)
        fcst_df = fcst_df.reset_index()
        re_renamer = {v: k for k, v in renamer.items()}
        re_renamer["target"] = self.alias
        fcst_df = fcst_df.rename(columns=re_renamer)
        if quantiles is None:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        else:
            q_renamer = {
                q_orig: f"{self.alias}-q-{int(100 * q_user)}"
                for q_orig, q_user in zip(
                    DEFAULT_QUANTILE_CONFIG,
                    quantiles,
                    strict=True,
                )
            }
            fcst_df = fcst_df.rename(columns=q_renamer)
        return pd.DataFrame(fcst_df)

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
        if qc.quantiles is not None and not np.allclose(
            qc.quantiles,
            DEFAULT_QUANTILE_CONFIG,
        ):
            raise ValueError(
                "TabPFN only supports the default quantiles, "
                "please use the default quantiles or default level, "
            )
        with self._get_model() as model:
            fcst_df = self._forecast(
                model,
                df,
                h,
                quantiles=qc.quantiles,
            )
        if qc.quantiles is not None:
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
