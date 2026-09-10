"""End-to-end benchmarks of the forecast / cross-validation / anomaly APIs.

The models used here are trivial by design (see `_models.py`): what is measured
is the library's own orchestration cost - validation, frequency inference,
backtest splitting, merging and residual statistics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ._models import ConstantModel, SeasonalNaiveModel
from foundationforecast import FoundationForecast


def test_forecast_single_model(benchmark, large_panel: pd.DataFrame):
    ff = FoundationForecast(models=[SeasonalNaiveModel()])
    fcst = benchmark(ff.forecast, large_panel, 12, "D")
    assert len(fcst) == 12 * large_panel["unique_id"].nunique()


def test_forecast_with_levels(benchmark, large_panel: pd.DataFrame):
    ff = FoundationForecast(models=[SeasonalNaiveModel()])
    fcst = benchmark(ff.forecast, large_panel, 12, "D", [80, 90])
    assert "SeasonalNaive-lo-90" in fcst.columns


def test_forecast_multi_model_merge(benchmark, large_panel: pd.DataFrame):
    ff = FoundationForecast(
        models=[ConstantModel(alias=f"Constant{i}", value=float(i)) for i in range(4)]
    )
    fcst = benchmark(ff.forecast, large_panel, 12, "D")
    assert "Constant3" in fcst.columns


def test_forecast_infer_freq_and_sort(benchmark, unsorted_panel: pd.DataFrame):
    """Forecast on shuffled input, exercising freq inference and sorting."""
    ff = FoundationForecast(models=[ConstantModel()])
    fcst = benchmark(ff.forecast, unsorted_panel, 12)
    assert len(fcst) == 12 * unsorted_panel["unique_id"].nunique()


@pytest.mark.parametrize("n_windows", [1, 4])
def test_cross_validation(
    benchmark,
    small_panel: pd.DataFrame,
    n_windows: int,
):
    ff = FoundationForecast(models=[SeasonalNaiveModel()])
    cv_df = benchmark(ff.cross_validation, small_panel, 12, "D", n_windows)
    assert len(cv_df) == 12 * n_windows * small_panel["unique_id"].nunique()


def test_detect_anomalies(benchmark, small_panel: pd.DataFrame):
    ff = FoundationForecast(models=[SeasonalNaiveModel()])
    anomalies = benchmark(ff.detect_anomalies, small_panel, 12, "D", 3)
    assert "SeasonalNaive-anomaly" in anomalies.columns
