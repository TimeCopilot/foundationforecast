from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from utilsforecast.processing import group_by_agg, make_future_dataframe, process_df


class PanelData(NamedTuple):
    uids: pd.Series | np.ndarray
    last_times: np.ndarray
    series_arrays: list[np.ndarray]


def process_panel_from_df(
    df: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> PanelData:
    processed = process_df(df, id_col, time_col, target_col)
    series_arrays = [
        processed.data[s:e, 0]
        for s, e in zip(processed.indptr[:-1], processed.indptr[1:], strict=True)
    ]
    return PanelData(processed.uids, processed.last_times, series_arrays)


def grouped_std_by_id(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = group_by_agg(df, id_col, {value_col: "std"})
    std_col = "residual_std" if value_col == "residuals" else f"{value_col}_std"
    return out.rename(columns={value_col: std_col})


class TimeSeriesDataset:
    def __init__(
        self,
        series_arrays: list[np.ndarray],
        uids: pd.Series | np.ndarray,
        last_times: np.ndarray,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._series_arrays = series_arrays
        self.uids = uids
        self.last_times = last_times
        self.batch_size = batch_size
        self.dtype = dtype
        self._tensors: list[torch.Tensor] | None = None
        self.n_batches = len(series_arrays) // self.batch_size + (
            0 if len(series_arrays) % self.batch_size == 0 else 1
        )
        self.current_batch = 0

    @property
    def data(self) -> list[torch.Tensor]:
        if self._tensors is None:
            self._tensors = [
                torch.as_tensor(arr, dtype=self.dtype) for arr in self._series_arrays
            ]
        return self._tensors

    @classmethod
    def from_panel(
        cls,
        panel: PanelData,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> TimeSeriesDataset:
        return cls(panel.series_arrays, panel.uids, panel.last_times, batch_size, dtype)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        panel: PanelData | None = None,
    ) -> TimeSeriesDataset:
        if panel is None:
            panel = process_panel_from_df(df)
        return cls.from_panel(panel, batch_size, dtype)

    def __len__(self):
        return self.n_batches

    def make_future_dataframe(self, h: int, freq: str) -> pd.DataFrame:
        return make_future_dataframe(
            uids=self.uids,
            last_times=pd.to_datetime(self.last_times),
            h=h,
            freq=freq,
        )  # type: ignore

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.n_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.current_batch += 1
            batch_arrays = self._series_arrays[start_idx:end_idx]
            return [torch.as_tensor(arr, dtype=self.dtype) for arr in batch_arrays]
        raise StopIteration
