import pytest

from foundationforecast import FoundationForecast
from foundationforecast.core.forecaster import Forecaster
from foundationforecast.core.utils import TimeSeriesDataset

pytestmark = pytest.mark.benchmark


class DatasetTouchingModel(Forecaster):
    """Lightweight model that exercises dataset construction like real forecasters."""

    batch_size = 32

    def __init__(self, alias: str = "DatasetTouch"):
        self.alias = alias

    def forecast(self, df, h, freq=None, level=None, quantiles=None, **kwargs):
        panel = kwargs.get("panel")
        ds_kwargs = {"df": df, "batch_size": self.batch_size}
        if panel is not None:
            ds_kwargs["panel"] = panel
        dataset = TimeSeriesDataset.from_df(**ds_kwargs)
        fcst = dataset.make_future_dataframe(h=h, freq=freq or "D")
        fcst[self.alias] = 1.0
        return fcst


@pytest.mark.parametrize("n_models", [3, 5], ids=["3-models", "5-models"])
def test_foundation_forecast_multi_model(benchmark, large_panel_df, n_models):
    models: list[Forecaster] = [
        DatasetTouchingModel(alias=f"DatasetTouch{i}") for i in range(n_models)
    ]
    forecaster = FoundationForecast(models=models)
    result = benchmark(forecaster.forecast, large_panel_df, h=12, freq="D")
    assert len(result) == large_panel_df["unique_id"].nunique() * 12
    for model in models:
        assert model.alias in result.columns
