import numpy as np
import torch

from tests.helpers import generate_series
from foundationforecast.core.utils import (
    TimeSeriesDataset,
    process_panel_from_df,
)


def test_timeseries_dataset_class_default_dtype_is_bfloat16():
    """Ensure TimeSeriesDataset defaults to bfloat16 for backward compatibility."""
    df = generate_series(1, "D", min_length=10, max_length=10)
    dataset = TimeSeriesDataset.from_df(df, batch_size=10)
    assert dataset.data[0].dtype == torch.bfloat16


def test_timeseries_dataset_respects_custom_dtype():
    """Ensure TimeSeriesDataset respects custom dtype parameter."""
    df = generate_series(1, "D", min_length=10, max_length=10)
    dataset = TimeSeriesDataset.from_df(df, batch_size=10, dtype=torch.float32)
    assert dataset.data[0].dtype == torch.float32


def test_timeseries_dataset_multi_series_order_and_values():
    df = generate_series(3, "D", min_length=5, max_length=5, seed=0)
    panel = process_panel_from_df(df)
    dataset = TimeSeriesDataset.from_panel(panel, batch_size=2, dtype=torch.float32)

    assert len(panel.series_arrays) == 3
    assert len(dataset.uids) == 3
    for arr, tensor in zip(panel.series_arrays, dataset.data, strict=True):
        np.testing.assert_allclose(arr, tensor.numpy())

    batches = list(dataset)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_timeseries_dataset_from_panel_matches_from_df():
    df = generate_series(5, "D", min_length=8, max_length=12, seed=1)
    panel = process_panel_from_df(df)
    from_df = TimeSeriesDataset.from_df(df, batch_size=3, dtype=torch.float32)
    from_panel = TimeSeriesDataset.from_panel(panel, batch_size=3, dtype=torch.float32)

    assert list(from_df.uids) == list(from_panel.uids)
    for left, right in zip(from_df.data, from_panel.data, strict=True):
        torch.testing.assert_close(left, right)


def test_lazy_batch_iteration():
    df = generate_series(4, "D", min_length=6, max_length=6, seed=2)
    dataset = TimeSeriesDataset.from_df(df, batch_size=2, dtype=torch.float32)
    assert dataset._tensors is None
    first_batch = next(iter(dataset))
    assert dataset._tensors is None
    assert len(first_batch) == 2
