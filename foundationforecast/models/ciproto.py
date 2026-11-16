from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..core.conformal import ConformalIntervals
from ..core.exog.covariates import ExogStrategyConfig
from ..core.forecaster import ExogCapableForecaster, QuantileConverter
from ..core.quantiles import resolve_quantile_values
from ..core.utils import PanelData, process_panel_from_df

if TYPE_CHECKING:
    from continuousintelligence.ci_proto import Forecaster

_INT64_PAD = np.iinfo(np.int64).min
_CI_INSTALL_MSG = (
    "CIProto requires the continuousintelligence package. "
    'Install with: pip install "foundationforecast[ci-proto]"'
)


def is_ciproto_available() -> bool:
    """Return whether the optional ``continuousintelligence`` extra is installed."""
    return importlib.util.find_spec("continuousintelligence") is not None


def _import_ci() -> Any:
    try:
        import continuousintelligence as ci
    except ImportError as exc:
        raise ImportError(_CI_INSTALL_MSG) from exc
    return ci


class CIProto(ExogCapableForecaster):
    """
      ci-proto is Continuous Intelligence's zero-shot probabilistic forecaster
    (~167M parameters) with a very long context (16k standard, longer via an
    automatic long checkpoint). It returns nine native quantiles between 0.1
    and 0.9 and supports known-future covariates. See the
    [official repo](https://github.com/continuousintelligence/ci) and
    [model card](https://huggingface.co/ContinuousIntelligence/ci-proto).

      Requires the optional extra ``foundationforecast[ci-proto]`` (PyPI package
      ``continuousintelligence``). Model weights use the
      [ci-proto Code and Model Licence](https://github.com/continuousintelligence/ci/blob/main/LICENCE).
    """

    def __init__(
        self,
        repo_id: str = "ContinuousIntelligence/ci-proto",
        revision: str | None = None,
        device: str = "auto",
        context_length: int | None = 16_384,
        batch_size: int = 8,
        alias: str = "CIProto",
        reuse_loaded_model: bool = True,
        exog_strategy: ExogStrategyConfig = "auto",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): Hugging Face model ID or local checkpoint
                bundle. Defaults to ``ContinuousIntelligence/ci-proto``.
            revision (str | None, optional): Hugging Face commit SHA for
                reproducible Hub loads. Defaults to None (latest).
            device (str, optional): ``"auto"``, ``"cpu"``, or ``"cuda"`` for
                upstream inference. Defaults to ``"auto"``.
            context_length (int | None, optional): Maximum history steps passed
                per series before padding. Defaults to 16_384 (standard
                checkpoint context). Longer histories trigger the long
                checkpoint automatically upstream.
            batch_size (int, optional): Panel series per ``series_ids`` cohort
                call. Defaults to 8.
            alias (str, optional): Output column name. Defaults to ``"CIProto"``.
            reuse_loaded_model (bool, optional): Cache weights across calls.
                Defaults to True.
            exog_strategy (ExogStrategyConfig, optional): How ``X_df`` is
                handled. Defaults to ``"auto"`` (native future covariates).
            prediction_intervals (ConformalIntervals | None, optional): When
                set, conformal intervals replace native quantiles. Defaults to
                None.

        Notes:
            **Installation:**

            - ``pip install "foundationforecast[ci-proto]"`` or
              ``uv add "foundationforecast[ci-proto]"``.

            **Technical details:**

            - Each panel batch is forecast as an independent univariate cohort
              via ``series_ids``.
            - ``level`` / ``quantiles`` use the native 0.1–0.9 knot grid with
              interpolation. {FIXED_KNOT_QUANTILES_NOTE}
            - Calendar features activate when ``freq`` is supplied and
              timestamps are built from ``ds``.
        """
        super().__init__(
            reuse_loaded_model=reuse_loaded_model,
            exog_strategy=exog_strategy,
            prediction_intervals=prediction_intervals,
        )
        if context_length is not None and context_length < 1:
            raise ValueError("`context_length` must be a positive integer or None.")
        self.repo_id = repo_id
        self.revision = revision
        self.device = device
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias

    def supports_native_futr_exog(self) -> bool:
        return True

    def _load_model(self) -> Forecaster:
        ci = _import_ci()
        kwargs: dict[str, Any] = {"device": self.device}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return ci.load_model(self.repo_id, **kwargs)

    def _model_cache_prefix(self) -> str | None:
        rev = self.revision or "main"
        return f"ciproto:{self.repo_id}:{rev}:{self.device}"

    @contextmanager
    def _get_model(self) -> Forecaster:
        with self._cached_model(self._load_model) as model:
            yield model

    @staticmethod
    def _tail(series: np.ndarray, context_length: int | None) -> np.ndarray:
        y = np.asarray(series, dtype=np.float64)
        if context_length is not None and len(y) > context_length:
            y = y[-context_length:]
        return y

    def _pad_histories(self, rows: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
        lengths = [len(r) for r in rows]
        width = max(lengths)
        hist = np.full((len(rows), width), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            hist[i, width - len(row) :] = row.astype(np.float32)
        pads = [width - length for length in lengths]
        return hist, pads

    @staticmethod
    def _ds_to_ns(ds: np.ndarray) -> np.ndarray:
        stamps = pd.to_datetime(ds).to_numpy(dtype="datetime64[ns]")
        return stamps.view(np.int64)

    def _batch_timestamps(
        self,
        df: pd.DataFrame,
        uids: list,
        left_pad: list[int],
        width: int,
        horizon_df: pd.DataFrame,
    ) -> np.ndarray:
        h = len(horizon_df[horizon_df["unique_id"] == uids[0]])
        total = width + h
        out = np.full((len(uids), total), _INT64_PAD, dtype=np.int64)
        for vi, uid in enumerate(uids):
            series = df.loc[df["unique_id"] == uid].sort_values("ds")
            if self.context_length is not None:
                series = series.iloc[-self.context_length :]
            hist_ns = self._ds_to_ns(series["ds"].to_numpy())
            fut_ds = horizon_df.loc[horizon_df["unique_id"] == uid, "ds"].to_numpy()
            fut_ns = self._ds_to_ns(fut_ds)
            pad = left_pad[vi]
            out[vi, pad : pad + len(hist_ns)] = hist_ns
            out[vi, width : width + len(fut_ns)] = fut_ns
        return out

    def _batch_future_covariates(
        self,
        df: pd.DataFrame,
        uids: list,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
        width: int,
        left_pad: list[int],
        h: int,
    ) -> torch.Tensor:
        n_cov = len(futr_exog_list)
        out = np.full((len(uids), n_cov, width + h), np.nan, dtype=np.float32)
        for vi, uid in enumerate(uids):
            series = df.loc[df["unique_id"] == uid].sort_values("ds")
            if self.context_length is not None:
                series = series.iloc[-self.context_length :]
            hist = series[futr_exog_list].to_numpy(dtype=np.float32).T
            fut = (
                horizon_df.loc[horizon_df["unique_id"] == uid, futr_exog_list]
                .to_numpy(dtype=np.float32)
                .T
            )
            block = np.concatenate([hist, fut], axis=1)
            pad = left_pad[vi]
            out[vi, :, pad : pad + block.shape[1]] = block
        return torch.tensor(out)

    def _native_quantile_list(self, model: Forecaster) -> list[float]:
        return [float(q) for q in model.quantiles.tolist()]

    def _forecast_batch(
        self,
        model: Forecaster,
        df: pd.DataFrame,
        uids: list,
        histories: list[np.ndarray],
        h: int,
        freq: str | None,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str] | None,
    ) -> np.ndarray:
        hist_np, left_pad = self._pad_histories(histories)
        width = hist_np.shape[1]
        history = torch.tensor(hist_np)
        kwargs: dict[str, Any] = {
            "horizon": h,
            "series_ids": tuple(str(uid) for uid in uids),
        }
        if freq is not None:
            kwargs["timestamps"] = torch.tensor(
                self._batch_timestamps(df, uids, left_pad, width, horizon_df)
            )
            kwargs["freq"] = freq
        if futr_exog_list:
            kwargs["future_covariates"] = self._batch_future_covariates(
                df, uids, horizon_df, futr_exog_list, width, left_pad, h
            )
        pred = model.forecast(history, **kwargs)
        return pred.numpy()

    def _run_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        qc: QuantileConverter,
        panel: PanelData,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str] | None,
    ) -> pd.DataFrame:
        fcst_df = horizon_df[["unique_id", "ds"]].copy()
        need_q = qc.quantiles is not None
        native_q: list[float] = []
        med_idx = 0

        points: list[np.ndarray] = []
        q_blocks: list[np.ndarray] = []

        uids = list(panel.uids)
        arrays = panel.series_arrays
        batches = range(0, len(arrays), self.batch_size)

        with self._get_model() as model:
            native_q = self._native_quantile_list(model)
            med_idx = native_q.index(0.5)
            for start in tqdm(batches, disable=len(batches) <= 1):
                stop = min(start + self.batch_size, len(arrays))
                batch_uids = uids[start:stop]
                batch_hist = [
                    self._tail(arrays[i], self.context_length)
                    for i in range(start, stop)
                ]
                pred = self._forecast_batch(
                    model,
                    df,
                    batch_uids,
                    batch_hist,
                    h,
                    freq,
                    horizon_df,
                    futr_exog_list,
                )
                # pred: [V, Q, H]
                points.append(pred[:, med_idx, :].reshape(-1))
                if need_q:
                    req = sorted({float(q) for q in qc.quantiles or []})
                    chosen = resolve_quantile_values(
                        native_q,
                        pred,
                        req,
                        axis=1,
                    )
                    q_blocks.append(
                        np.transpose(chosen, (0, 2, 1)).reshape(-1, len(req))
                    )

        fcst_df[self.alias] = np.concatenate(points)
        if need_q and qc.quantiles is not None:
            q_all = np.concatenate(q_blocks, axis=0)
            fcst_df = self._assign_quantile_forecasts(
                fcst_df,
                self.alias,
                qc.quantiles,
                q_all,
            )
            fcst_df = qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])
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
        if panel is None:
            panel = process_panel_from_df(df[["unique_id", "ds", "y"]])
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        return self._run_forecast(
            df,
            h,
            freq,
            qc,
            panel,
            horizon_df,
            futr_exog_list,
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
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = self._make_timeseries_dataset(
            df,
            batch_size=self.batch_size,
            dtype=torch.float32,
            panel=panel,
        )
        if panel is None:
            panel = process_panel_from_df(df)
        horizon_df = dataset.make_future_dataframe(h=h, freq=freq)
        return self._run_forecast(df, h, freq, qc, panel, horizon_df, None)
