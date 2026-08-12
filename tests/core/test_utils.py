import torch

from tests.helpers import generate_series
from foundationforecast.core.utils import TimeSeriesDataset


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
