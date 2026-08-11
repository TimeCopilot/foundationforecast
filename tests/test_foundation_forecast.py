import pandas as pd
import pytest

from foundationforecast import FoundationForecast
from foundationforecast.core.forecaster import Forecaster


class DummyModel(Forecaster):
    alias = "dummy"

    def forecast(self, df, h, freq=None, level=None, quantiles=None):
        out = df[["unique_id", "ds"]].drop_duplicates("unique_id").copy()
        out = out.assign(ds=pd.Timestamp("2020-01-01"))
        out = pd.concat([out.assign(ds=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)) for i in range(h)])
        out["dummy"] = 1.0
        return out[["unique_id", "ds", "dummy"]]


def test_foundation_forecast_requires_unique_aliases():
    with pytest.raises(ValueError, match="Duplicate model aliases"):
        FoundationForecast(models=[DummyModel(), DummyModel()])


def test_foundation_forecast_has_forecast_method():
    ff = FoundationForecast(models=[DummyModel()])
    assert hasattr(ff, "forecast")
