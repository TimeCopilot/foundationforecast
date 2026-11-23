from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from simons import QUANTILES
from simons import Simons as SimonsModel
from tqdm import tqdm

from ..core.conformal import ConformalIntervals
from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.quantiles import resolve_quantile_values
from ..core.utils import PanelData

_NATIVE_LEVELS = [float(level) for level in QUANTILES]


class Simons(ExogCapableForecaster):
    """
    Simons is Abel AI Lab's family of open time series foundation models for
    zero-shot probabilistic forecasting. It is a causal patch transformer
    (patch size 32) with attention across variates, native known-future
    covariates, and nine quantile outputs (0.1–0.9). See the
    [model collection](https://huggingface.co/collections/abel-lab/simons-1-6ae3d4ce9705938e293c52dd)
    on Hugging Face.
    """

    def __init__(
        self,
        repo_id: str = "abel-lab/Simons-297M",
        context_length: int = 32768,
        batch_size: int = 64,
        alias: str = "Simons",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): Hugging Face Hub model ID or a local
                directory with ``config.json`` and ``model.safetensors``.
                Defaults to ``abel-lab/Simons-297M``.
            context_length (int, optional): Maximum history length passed to
                the model. Each checkpoint crops further to its own context
                (16,384 for 25M and 60M, 32,768 for 178M and 297M). Defaults
                to 32768.
            batch_size (int, optional): Series per ``forecast()`` batch.
                Defaults to 64.
            alias (str, optional): Column prefix in output DataFrames.
                Defaults to ``Simons``.
            reuse_loaded_model (bool, optional): Reuse loaded weights across
                calls via the process-wide cache. Defaults to True.
            exog_strategy (ExogStrategyConfig, optional): How ``X_df`` is
                handled. Defaults to ``"auto"`` (native known-future covariates).
            prediction_intervals (ConformalIntervals | None, optional): When set,
                ``forecast()`` replaces native quantiles with conformal intervals
                calibrated on cross-validation errors of the point forecast. See
                the conformal prediction docs. Defaults to None.

        Notes:
            **Requirements:** ``simons>=0.1.2`` from
            [Abel-ai-lab/Simons](https://github.com/Abel-ai-lab/Simons)
            (Python 3.10+, PyTorch 2.5+). The package is not on PyPI; this
            library pins git tag ``v0.1.2``.

            **Available models:**

            | Model ID | Parameters | Context |
            | -------- | ---------- | ------- |
            | [`abel-lab/Simons-25M`](https://huggingface.co/abel-lab/Simons-25M) | ~25M | 16,384 |
            | [`abel-lab/Simons-60M`](https://huggingface.co/abel-lab/Simons-60M) | ~60M | 16,384 |
            | [`abel-lab/Simons-178M`](https://huggingface.co/abel-lab/Simons-178M) | ~178M | 32,768 |
            | [`abel-lab/Simons-297M`](https://huggingface.co/abel-lab/Simons-297M) | ~297M | 32,768 |

            The point forecast is the 0.5 quantile. Horizons longer than the
            checkpoint's 512-step block are rolled out inside ``simons``.
            Non-negative histories are clipped at zero and integer-valued
            histories are rounded, following the checkpoint config. Known-future
            covariates are numeric columns in ``X_df`` (up to 16 usable
            columns; constant columns are dropped).

            **License:** checkpoint weights use OpenMDW-1.1. The inference code
            is Apache-2.0.
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> SimonsModel:
        return SimonsModel.from_pretrained(self.repo_id, device=self.device)

    @contextmanager
    def _get_model(self) -> SimonsModel:
        with self._cached_model(self._load_model) as model:
            yield model

    def _batch_width(self, batch: list[torch.Tensor]) -> int:
        return min(max(int(ts.shape[0]) for ts in batch), self.context_length)

    def _to_history(self, batch: list[torch.Tensor]) -> np.ndarray:
        """Build ``(B, T)`` arrays with NaN left-padding."""
        width = self._batch_width(batch)
        history = np.full((len(batch), width), np.nan, dtype=np.float32)
        for idx, ts in enumerate(batch):
            values = ts[-width:].detach().to(dtype=torch.float32).cpu().numpy()
            history[idx, -len(values) :] = values
        return history

    def _to_covariates(
        self,
        batch: list[torch.Tensor],
        uids: list | np.ndarray,
        df: pd.DataFrame,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
        h: int,
    ) -> np.ndarray:
        """Build ``(B, C, T + h)`` arrays aligned with ``_to_history``."""
        width = self._batch_width(batch)
        covariates = np.full(
            (len(batch), len(futr_exog_list), width + h),
            np.nan,
            dtype=np.float32,
        )
        for idx, uid in enumerate(uids):
            hist = df.loc[df["unique_id"] == uid, futr_exog_list].to_numpy(
                dtype=np.float32
            )
            if hist.shape[0] > width:
                hist = hist[-width:]
            horizon = horizon_df.loc[
                horizon_df["unique_id"] == uid, futr_exog_list
            ].to_numpy(dtype=np.float32)
            if horizon.shape[0] != h:
                raise ValueError("Horizon exog length mismatch.")
            n_hist = hist.shape[0]
            covariates[idx, :, width - n_hist : width] = hist.T
            covariates[idx, :, width:] = horizon.T
        return covariates

    def supports_native_futr_exog(self) -> bool:
        return True

    def _forecast_batch(
        self,
        model: SimonsModel,
        batch: list[torch.Tensor],
        h: int,
        covariates: np.ndarray | None = None,
    ) -> np.ndarray:
        kwargs: dict[str, np.ndarray] = {}
        if covariates is not None:
            kwargs["covariates"] = covariates
        out = model.forecast(self._to_history(batch), horizon=h, **kwargs)
        if isinstance(out, torch.Tensor):
            out = out.detach().to(dtype=torch.float32).cpu().numpy()
        return np.asarray(out, dtype=np.float32)

    def _assign_native_quantiles(
        self,
        fcst_df: pd.DataFrame,
        quantiles: np.ndarray,
        level: list[int | float] | None,
        requested: list[float] | None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=requested)
        median_idx = _NATIVE_LEVELS.index(0.5)
        fcst_df[self.alias] = quantiles[..., median_idx].reshape(-1, 1)
        if qc.quantiles is None:
            return fcst_df
        resolved = resolve_quantile_values(
            _NATIVE_LEVELS,
            quantiles,
            qc.quantiles,
            axis=-1,
        )
        fcst_df = self._assign_quantile_forecasts(
            fcst_df,
            self.alias,
            qc.quantiles,
            resolved,
        )
        return qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])

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
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, panel=panel
        )
        fcst_df = horizon_df[["unique_id", "ds"]].copy()
        forecasts: list[np.ndarray] = []
        uids = np.asarray(dataset.uids)
        with self._get_model() as model:
            for batch in tqdm(dataset):
                batch_len = len(batch)
                start = (dataset.current_batch - 1) * dataset.batch_size
                covariates = self._to_covariates(
                    batch,
                    uids[start : start + batch_len],
                    df,
                    horizon_df,
                    futr_exog_list,
                    h,
                )
                forecasts.append(self._forecast_batch(model, batch, h, covariates))
        return self._assign_native_quantiles(
            fcst_df, np.concatenate(forecasts, axis=0), level, quantiles
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
        freq = self._maybe_infer_freq(df, freq)
        dataset = self._make_timeseries_dataset(
            df, batch_size=self.batch_size, panel=panel
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        forecasts: list[np.ndarray] = []
        with self._get_model() as model:
            for batch in tqdm(dataset):
                forecasts.append(self._forecast_batch(model, batch, h))
        return self._assign_native_quantiles(
            fcst_df, np.concatenate(forecasts, axis=0), level, quantiles
        )
