import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.forecaster import Forecaster
from foundationforecast.core.utils import TimeSeriesDataset, process_panel_from_df

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize(
    "panel_df_fixture",
    ["panel_df", "large_panel_df"],
    ids=["small", "large"],
)
def test_process_panel_from_df(benchmark, panel_df_fixture, request):
    df = request.getfixturevalue(panel_df_fixture)
    panel = benchmark(process_panel_from_df, df)
    assert len(panel.series_arrays) == df["unique_id"].nunique()


def test_process_panel_from_df_string_ds(benchmark, string_ds_large_panel_df):
    panel = benchmark(process_panel_from_df, string_ds_large_panel_df)
    assert len(panel.series_arrays) == string_ds_large_panel_df["unique_id"].nunique()


def test_timeseries_dataset_from_panel(benchmark, large_panel_data):
    dataset = benchmark(
        TimeSeriesDataset.from_panel,
        large_panel_data,
        32,
    )
    assert len(dataset) > 0


def test_timeseries_dataset_from_df_with_panel(
    benchmark,
    large_panel_df,
    large_panel_data,
):
    def build_dataset():
        return TimeSeriesDataset.from_df(
            large_panel_df,
            batch_size=32,
            panel=large_panel_data,
        )

    dataset = benchmark(build_dataset)
    assert len(dataset) > 0


def test_assign_quantile_forecasts(benchmark):
    n_rows = 120
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fcst_df = pd.DataFrame(index=range(n_rows))
    fcsts_quantiles_np = np.ones((10, 12, len(quantiles)))

    def assign():
        return Forecaster._assign_quantile_forecasts(
            fcst_df,
            "Model",
            quantiles,
            fcsts_quantiles_np,
        )

    out = benchmark(assign)
    assert len(out.columns) == len(quantiles)
