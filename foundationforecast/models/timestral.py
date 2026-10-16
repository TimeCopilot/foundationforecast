from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster
from tqdm import tqdm

from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.quantiles import select_clipped_quantile_values
from ..core.utils import PanelData

TIMESTRAL_QUANTILE_RANGE = (0.01, 0.99)


class Timestral(ExogCapableForecaster):
    """
    Timestral is Mistral AI's family of time series foundation models for
    zero-shot probabilistic forecasting. It is a causal patch-based
    transformer with decoupled horizon forecasting, native support for
    known-future and past-only covariates, and 21 quantile outputs
    (0.01–0.99). See the
    [model collection](https://huggingface.co/collections/mistralai/timestral)
    on Hugging Face.
    """

    def __init__(
        self,
        repo_id: str = "mistralai/Timestral-209M",
        context_length: int = 32768,
        batch_size: int = 64,
        alias: str = "Timestral",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): Hugging Face Hub model ID. Defaults to
                ``mistralai/Timestral-209M``.
            context_length (int, optional): Maximum history length passed to
                the model. Defaults to 32768.
            batch_size (int, optional): Series per ``forecast()`` batch.
                Defaults to 64.
            alias (str, optional): Column prefix in output DataFrames.
                Defaults to ``Timestral``.
            reuse_loaded_model (bool, optional): Reuse loaded weights across
                calls via the process-wide cache. Defaults to True.
            exog_strategy (ExogStrategyConfig, optional): How ``X_df`` is
                handled. Defaults to ``"auto"`` (native known-future covariates).
            symmetric_averaging (bool, optional): Average each forecast with
                the mirrored forecast of the negated series. Defaults to True,
                matching the published benchmark settings.
            clip_nonnegative (bool, optional): After forecasting, clamp every
                quantile to ≥ 0 when that series' observed history is entirely
                non-negative. Defaults to True.

        Notes:
            **Requirements:** ``timecopilot-timestral>=0.1.0`` (Python 3.10+).

            **Available models:**

            | Model ID | Parameters |
            | -------- | ---------- |
            | [`mistralai/Timestral-5M`](https://huggingface.co/mistralai/Timestral-5M) | ~5M |
            | [`mistralai/Timestral-28M`](https://huggingface.co/mistralai/Timestral-28M) | ~28M |
            | [`mistralai/Timestral-123M`](https://huggingface.co/mistralai/Timestral-123M) | ~123M |
            | [`mistralai/Timestral-209M`](https://huggingface.co/mistralai/Timestral-209M) | ~209M |

            **License:** checkpoint weights use Mistral's Modified MIT license
            (revenue cap for companies above USD 20M monthly revenue). The
            inference code is Apache-2.0 via ``timecopilot-timestral``.
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
        )
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> TimestralForecaster:
        return TimestralForecaster.from_pretrained(
            self.repo_id,
            device=self.device,
            max_context=self.context_length,
        )

    @contextmanager
    def _get_model(self) -> TimestralForecaster:
        with self._cached_model(self._load_model) as model:
            yield model

    def _batch_context_width(self, batch: list[torch.Tensor]) -> int:
        return min(max(len(ts) for ts in batch), self.context_length)

    def _to_context(self, batch: list[torch.Tensor]) -> np.ndarray:
        """Build ``(B, 1, T)`` arrays with NaN left-padding."""
        max_len = self._batch_context_width(batch)
        context = np.full((len(batch), 1, max_len), np.nan, dtype=np.float32)
        for idx, ts in enumerate(batch):
            values = ts[-max_len:].to(dtype=torch.float32).numpy()
            context[idx, 0, -len(values) :] = values
        return context

    def _to_known_covariates(
        self,
        batch: list[torch.Tensor],
        uids: list | np.ndarray,
        df: pd.DataFrame,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
        h: int,
    ) -> np.ndarray:
        """Build ``(B, C, T + h)`` arrays for Timestral ``known_covariates``."""
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

    @staticmethod
    def _quantiles_numpy(out) -> np.ndarray:
        quantiles = out.quantiles
        if isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()
        return quantiles[:, 0, :, :]

    @staticmethod
    def _median_numpy(out) -> np.ndarray:
        median = out.median
        if isinstance(median, torch.Tensor):
            median = median.cpu().numpy()
        return median[:, 0, :]

    def _inference_kwargs(self) -> dict[str, bool]:
        return {
            "symmetric_averaging": self.symmetric_averaging,
            "clip_nonnegative": self.clip_nonnegative,
        }

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
        q_min, q_max = TIMESTRAL_QUANTILE_RANGE
        fcsts: list[np.ndarray] = []
        native_levels: list[float] | None = None
        uids = np.asarray(dataset.uids)
        with self._get_model() as model:
            for batch in tqdm(dataset):
                batch_len = len(batch)
                start = (dataset.current_batch - 1) * dataset.batch_size
                batch_uids = uids[start : start + batch_len]
                known = self._to_known_covariates(
                    batch,
                    batch_uids,
                    df,
                    horizon_df,
                    futr_exog_list,
                    h,
                )
                out = model.forecast(
                    self._to_context(batch),
                    prediction_length=h,
                    known_covariates=known,
                    **self._inference_kwargs(),
                )
                if native_levels is None:
                    native_levels = [float(x) for x in out.quantile_levels]
                fcsts.append(self._quantiles_numpy(out))
        fcsts_np = np.concatenate(fcsts, axis=0)
        assert native_levels is not None
        median_idx = native_levels.index(0.5)
        fcst_df[self.alias] = fcsts_np[..., median_idx].reshape(-1, 1)
        if qc.quantiles is not None:
            fcsts_quantiles_np = select_clipped_quantile_values(
                native_levels,
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
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, panel=panel
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        q_min, q_max = TIMESTRAL_QUANTILE_RANGE
        fcsts: list[np.ndarray] = []
        native_levels: list[float] | None = None
        with self._get_model() as model:
            for batch in tqdm(dataset):
                out = model.forecast(
                    self._to_context(batch),
                    prediction_length=h,
                    **self._inference_kwargs(),
                )
                if native_levels is None:
                    native_levels = [float(x) for x in out.quantile_levels]
                fcsts.append(self._quantiles_numpy(out))
        fcsts_np = np.concatenate(fcsts, axis=0)
        assert native_levels is not None
        median_idx = native_levels.index(0.5)
        fcst_df[self.alias] = fcsts_np[..., median_idx].reshape(-1, 1)
        if qc.quantiles is not None:
            fcsts_quantiles_np = select_clipped_quantile_values(
                native_levels,
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
