"""Benchmarks for the tensor-side data preparation used by every model."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from foundationforecast.core.forecaster import _DataProcessor
from foundationforecast.core.utils import TimeSeriesDataset


@pytest.mark.parametrize("batch_size", [16, 128])
def test_time_series_dataset_from_df(
    benchmark,
    large_panel: pd.DataFrame,
    batch_size: int,
):
    dataset = benchmark(TimeSeriesDataset.from_df, large_panel, batch_size)
    assert len(dataset) > 0


def test_time_series_dataset_iteration(benchmark, large_panel: pd.DataFrame):
    dataset = TimeSeriesDataset.from_df(large_panel, batch_size=16)

    def consume() -> int:
        return sum(len(batch) for batch in dataset)

    assert benchmark(consume) == len(large_panel["unique_id"].unique())


def test_time_series_dataset_make_future_dataframe(
    benchmark,
    large_panel: pd.DataFrame,
):
    dataset = TimeSeriesDataset.from_df(large_panel, batch_size=32)
    future = benchmark(dataset.make_future_dataframe, 24, "D")
    assert len(future) == 24 * len(large_panel["unique_id"].unique())


def _variable_length_context(
    n_series: int = 128,
    max_len: int = 512,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [
        torch.rand(max_len - (i % 64), generator=generator, dtype=torch.float32)
        for i in range(n_series)
    ]


def test_data_processor_left_pad_and_stack(benchmark):
    processor = _DataProcessor(dtype=torch.float32, device=torch.device("cpu"))
    context = _variable_length_context()
    out = benchmark(processor._prepare_and_validate_context, context)
    assert out.shape[0] == len(context)


def test_data_processor_impute_missing(benchmark):
    processor = _DataProcessor(dtype=torch.float32, device=torch.device("cpu"))
    batch = processor._prepare_and_validate_context(_variable_length_context())
    out = benchmark(processor._maybe_impute_missing, batch)
    assert not torch.isnan(out).any()
