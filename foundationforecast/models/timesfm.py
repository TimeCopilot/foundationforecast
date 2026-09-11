import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import timesfm_v1
import torch
from huggingface_hub import repo_exists
from timesfm import TimesFM_2p5_200M_torch
from timesfm3 import ModelConfig, TimesFM3Evaluator
from timesfm_v1.timesfm_base import DEFAULT_QUANTILES as DEFAULT_QUANTILES_TFM
from tqdm import tqdm

from ..core.forecaster import (
    Forecaster,
    QuantileConverter,
    maybe_convert_col_to_datetime,
)
from ..core.utils import TimeSeriesDataset

# Legacy HF repo IDs from GIFT-Eval submissions without "pytorch" in the name.
# Still loaded via timesfm_v1 PyTorch checkpoints (JAX is not supported).
_GIFT_EVAL_LEGACY_REPOS = (
    "google/timesfm-1.0-200m",
    "google/timesfm-2.0-500m-jax",
)


class _TimesFMV1(Forecaster):
    def __init__(
        self,
        repo_id: str,
        context_length: int,
        batch_size: int,
        alias: str,
        reuse_loaded_model: bool = True,
    ):
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias

    def _model_cache_prefix(self) -> str:
        return f"timesfm_v1:{self.repo_id}"

    def _model_cache_key(
        self,
        prediction_length: int,
        quantiles: list[float] | None,
    ) -> str:
        quantiles_key = tuple(quantiles) if quantiles is not None else None
        v2_version = "2.0" in self.repo_id
        context_len = (
            min(self.context_length, 512) if not v2_version else self.context_length
        )
        return (
            f"{self._model_cache_prefix()}:"
            f"{context_len}:{self.batch_size}:{prediction_length}:{quantiles_key}"
        )

    def _load_predictor(
        self,
        prediction_length: int,
        quantiles: list[float] | None,
    ) -> timesfm_v1.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        # these values are based on
        # https://github.com/google-research/timesfm/blob/ba034ae71c2fc88eaf59f80b4a778cc2c0dca7d6/experiments/extended_benchmarks/run_timesfm.py#L91
        v2_version = "2.0" in self.repo_id
        context_len = (
            min(self.context_length, 512) if not v2_version else self.context_length
        )
        num_layers = 50 if v2_version else 20
        use_positional_embedding = not v2_version

        tfm_hparams = timesfm_v1.TimesFmHparams(
            backend=backend,
            horizon_len=prediction_length,
            quantiles=quantiles,
            context_len=context_len,
            num_layers=num_layers,
            use_positional_embedding=use_positional_embedding,
            per_core_batch_size=self.batch_size,
        )
        if os.path.exists(self.repo_id):
            path = os.path.join(self.repo_id, "torch_model.ckpt")
            tfm_checkpoint = timesfm_v1.TimesFmCheckpoint(path=path)
            return timesfm_v1.TimesFm(
                hparams=tfm_hparams,
                checkpoint=tfm_checkpoint,
            )
        if repo_exists(self.repo_id):
            tfm_checkpoint = timesfm_v1.TimesFmCheckpoint(
                huggingface_repo_id=self.repo_id
            )
            return timesfm_v1.TimesFm(
                hparams=tfm_hparams,
                checkpoint=tfm_checkpoint,
            )
        raise OSError(
            f"Failed to load model. Searched for '{self.repo_id}' "
            "as a local path to model directory and as a Hugging Face repo_id."
        )

    @contextmanager
    def _get_predictor(
        self,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> timesfm_v1.TimesFm:
        cache_key = self._model_cache_key(prediction_length, quantiles)

        def loader() -> timesfm_v1.TimesFm:
            return self._load_predictor(prediction_length, quantiles)

        with self._cached_model(loader, cache_key=cache_key) as tfm:
            yield tfm

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_datetime(df, "ds")
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        with self._get_predictor(
            prediction_length=h,
            quantiles=qc.quantiles or DEFAULT_QUANTILES_TFM,
        ) as predictor:
            fcst_df = predictor.forecast_on_df(
                inputs=df,
                freq=freq,
                value_name="y",
                model_name=self.alias,
                num_jobs=1,
            )
        if qc.quantiles is not None:
            renamer = {
                f"{self.alias}-q-{q}": f"{self.alias}-q-{int(q * 100)}"
                for q in qc.quantiles
            }
            fcst_df = fcst_df.rename(columns=renamer)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        else:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df


class _TimesFMV2_p5(Forecaster):
    def __init__(
        self,
        repo_id: str,
        context_length: int,
        batch_size: int,
        alias: str,
        reuse_loaded_model: bool = True,
        **kwargs: Any,
    ):
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.kwargs = kwargs

    def _model_cache_prefix(self) -> str:
        return f"timesfm_v2p5:{self.repo_id}"

    def _model_cache_key(self, prediction_length: int) -> str:
        kwargs_key = tuple(sorted((self.kwargs or {}).items()))
        return (
            f"{self._model_cache_prefix()}:"
            f"{self.context_length}:{prediction_length}:{kwargs_key}"
        )

    def _load_predictor(self, prediction_length: int) -> TimesFM_2p5_200M_torch:
        if os.path.exists(self.repo_id) or repo_exists(self.repo_id):
            tfm = TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
        else:
            raise OSError(
                f"Failed to load model. Searched for '{self.repo_id}' "
                "as a local path to model directory and as a Hugging Face repo_id."
            )
        default_kwargs = {
            "max_context": self.context_length,
            "max_horizon": prediction_length,
            "normalize_inputs": True,
            "use_continuous_quantile_head": True,
            "fix_quantile_crossing": True,
        }
        passed_kwargs = self.kwargs or {}
        compile_kwargs = {**default_kwargs, **passed_kwargs}
        config = timesfm.ForecastConfig(**compile_kwargs)
        tfm.compile(config)
        return tfm

    @contextmanager
    def _get_predictor(
        self,
        prediction_length: int,
    ) -> TimesFM_2p5_200M_torch:
        cache_key = self._model_cache_key(prediction_length)

        def loader() -> TimesFM_2p5_200M_torch:
            return self._load_predictor(prediction_length)

        with self._cached_model(loader, cache_key=cache_key) as tfm:
            yield tfm

    def _predict(
        self,
        model: TimesFM_2p5_200M_torch,
        dataset: TimeSeriesDataset,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fcsts = [
            model.forecast(
                inputs=batch,
                horizon=h,
            )
            for batch in tqdm(dataset)
        ]
        fcsts_mean, fcsts_quantiles = zip(*fcsts, strict=False)
        fcsts_mean_np = np.concatenate(fcsts_mean)
        fcsts_quantiles_np = np.concatenate(fcsts_quantiles)
        return fcsts_mean_np, fcsts_quantiles_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=torch.float32,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_predictor(prediction_length=h) as model:
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i + 1  # skip the first quantile (mean)
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df


class _TimesFMV3(Forecaster):
    def __init__(
        self,
        repo_id: str,
        context_length: int,
        batch_size: int,
        alias: str,
        reuse_loaded_model: bool = True,
        **kwargs: Any,
    ):
        super().__init__(reuse_loaded_model=reuse_loaded_model)
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.kwargs = kwargs

    def _model_cache_prefix(self) -> str:
        return f"timesfm_v3:{self.repo_id}"

    def _model_cache_key(self, prediction_length: int) -> str:
        del prediction_length
        kwargs_key = tuple(sorted((self.kwargs or {}).items()))
        return (
            f"{self._model_cache_prefix()}:"
            f"{self.context_length}:{self.batch_size}:{kwargs_key}"
        )

    def _load_predictor(self, prediction_length: int) -> TimesFM3Evaluator:
        if os.path.exists(self.repo_id) or repo_exists(self.repo_id):
            config = ModelConfig(
                checkpoint_path=self.repo_id,
                per_core_batch_size=self.batch_size,
                **(self.kwargs or {}),
            )
            return TimesFM3Evaluator(config)
        raise OSError(
            f"Failed to load model. Searched for '{self.repo_id}' "
            "as a local path to model directory and as a Hugging Face repo_id."
        )

    @contextmanager
    def _get_predictor(self, prediction_length: int) -> TimesFM3Evaluator:
        cache_key = self._model_cache_key(prediction_length)

        def loader() -> TimesFM3Evaluator:
            return self._load_predictor(prediction_length)

        with self._cached_model(loader, cache_key=cache_key) as forecaster:
            yield forecaster

    def _series_to_context(self, series: torch.Tensor) -> np.ndarray:
        arr = series.numpy().astype(np.float32)
        if len(arr) > self.context_length:
            arr = arr[-self.context_length :]
        return arr

    def _predict(
        self,
        forecaster: TimesFM3Evaluator,
        dataset: TimeSeriesDataset,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fcsts_mean: list[np.ndarray] = []
        fcsts_quantiles: list[np.ndarray] = []
        for batch in tqdm(dataset):
            contexts = [self._series_to_context(series) for series in batch]
            outputs = list(
                forecaster.predict_batch(
                    contexts=contexts,
                    horizon=h,
                    return_quantiles=True,
                    use_symmetric_averaging=True,
                    make_positive=True,
                    sort_quantiles=True,
                )
            )
            fcsts_mean.extend(output.forecast for output in outputs)
            fcsts_quantiles.extend(output.quantiles for output in outputs)
        fcsts_mean_np = np.stack(fcsts_mean)
        fcsts_quantiles_np = np.stack(fcsts_quantiles)
        return fcsts_mean_np, fcsts_quantiles_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=torch.float32,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_predictor(prediction_length=h) as forecaster:
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                forecaster,
                dataset,
                h,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df


class TimesFM(Forecaster):
    """
    TimesFM is a large time series model for time series forecasting, supporting both
    probabilistic and point forecasts. See the [official repo](https://github.com/
    google-research/timesfm) for more details.
    """

    def __new__(
        cls,
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 2048,
        batch_size: int = 64,
        alias: str = "TimesFM",
        reuse_loaded_model: bool = True,
        **kwargs: Any,
    ):
        if "pytorch" not in repo_id and repo_id not in _GIFT_EVAL_LEGACY_REPOS:
            legacy = ", ".join(_GIFT_EVAL_LEGACY_REPOS)
            raise ValueError(
                "TimesFM requires a PyTorch checkpoint repo_id (name contains "
                f"'pytorch') or a supported legacy GIFT-Eval repo_id: {legacy}. "
                "JAX backends are not supported."
            )
        if "1.0" in repo_id or "2.0" in repo_id:
            return _TimesFMV1(
                repo_id=repo_id,
                context_length=context_length,
                batch_size=batch_size,
                alias=alias,
                reuse_loaded_model=reuse_loaded_model,
            )
        elif "2.5" in repo_id:
            return _TimesFMV2_p5(
                repo_id=repo_id,
                context_length=context_length,
                batch_size=batch_size,
                alias=alias,
                reuse_loaded_model=reuse_loaded_model,
                **kwargs,
            )
        elif "3.0" in repo_id:
            return _TimesFMV3(
                repo_id=repo_id,
                context_length=context_length,
                batch_size=batch_size,
                alias=alias,
                reuse_loaded_model=reuse_loaded_model,
                **kwargs,
            )
        else:
            raise ValueError(
                "TimesFM only supports 1.0, 2.0, 2.5 and 3.0 models, please use a "
                "valid model id"
            )

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 2048,
        batch_size: int = 64,
        alias: str = "TimesFM",
        reuse_loaded_model: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the TimesFM model from. Examples include
                `google/timesfm-2.0-500m-pytorch`. Defaults to
                `google/timesfm-2.0-500m-pytorch`. See the full list of models at
                [Hugging Face](https://huggingface.co/collections/google/timesfm-release-
                66e4be5fdb56e960c1e482a6). Supported models:

                - `google/timesfm-1.0-200m-pytorch`
                - `google/timesfm-2.0-500m-pytorch`
                - `google/timesfm-2.5-200m-pytorch`
                - `google/timesfm-3.0-pytorch`
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 2048. For TimesFM 2.0 models, max is 2048
                (must be a multiple of 32). For TimesFM 1.0 models, max is 512. For
                TimesFM 3.0 models, max is 15360. See
                [TimesFM docs](https://github.com/google-research/timesfm#loading-the-
                model) for details.
            batch_size (int, optional): Batch size for inference. Defaults to 64.
                Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to `TimesFM`.
            **kwargs (Any): Extra keyword arguments forwarded to the backend
                model config. Used for TimesFM 2.5 and 3.0 models.

        Notes:
            **Academic Reference:**

            - Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)

            **Resources:**

            - GitHub: [google-research/timesfm](https://github.com/google-research/timesfm)
            - HuggingFace: [google/timesfm-release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)

            **Technical Details:**

            - Only PyTorch checkpoints are currently supported. JAX is not supported.
            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
            - TimesFM 3.0 pretrained weights are distributed under a
              [non-commercial license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
              and are restricted to non-commercial, non-production use.

            **Supported Models:**

            - `google/timesfm-1.0-200m-pytorch`
            - `google/timesfm-2.0-500m-pytorch`
            - `google/timesfm-2.5-200m-pytorch`
            - `google/timesfm-3.0-pytorch`
        """
        pass
