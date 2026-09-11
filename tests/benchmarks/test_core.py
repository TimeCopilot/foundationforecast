import pytest
from tests.helpers import SeasonalNaiveModel

from foundationforecast.core.forecaster import QuantileConverter, maybe_infer_freq
from foundationforecast.core.utils import TimeSeriesDataset

pytestmark = pytest.mark.benchmark


def test_maybe_infer_freq(benchmark, panel_df):
    result = benchmark(maybe_infer_freq, panel_df, None)
    assert result == "D"


def test_timeseries_dataset_from_df(benchmark, panel_df):
    def build_dataset():
        return TimeSeriesDataset.from_df(panel_df, batch_size=4)

    dataset = benchmark(build_dataset)
    assert len(dataset) > 0
    assert len(next(iter(dataset))) <= 4


def test_quantile_converter_level_to_quantiles(benchmark):
    def convert():
        qc = QuantileConverter(level=[80, 95])
        return qc.quantiles

    quantiles = benchmark(convert)
    assert quantiles is not None
    assert len(quantiles) > 0


def test_seasonal_naive_forecast(benchmark, panel_df):
    model = SeasonalNaiveModel()
    result = benchmark(model.forecast, panel_df, h=12, freq="D")
    assert len(result) == panel_df["unique_id"].nunique() * 12
    assert "SeasonalNaive" in result.columns
