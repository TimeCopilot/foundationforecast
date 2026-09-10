"""Benchmarks for the pure-python helpers used on every forecasting call."""

from __future__ import annotations

import pandas as pd
import pytest

from foundationforecast.core.forecaster import (
    Forecaster,
    QuantileConverter,
    get_seasonality,
    maybe_convert_col_to_datetime,
    maybe_infer_freq,
)
from foundationforecast.core.gluonts_forecaster import (
    maybe_convert_col_to_float32,
)


def test_validate_input(benchmark, large_panel: pd.DataFrame):
    benchmark(Forecaster.validate_input, large_panel, 12)


def test_maybe_infer_freq(benchmark, large_panel: pd.DataFrame):
    freq = benchmark(maybe_infer_freq, large_panel, None)
    assert freq == "D"


def test_maybe_convert_col_to_datetime_from_string(
    benchmark,
    string_ds_panel: pd.DataFrame,
):
    out = benchmark(maybe_convert_col_to_datetime, string_ds_panel, "ds")
    assert pd.api.types.is_datetime64_any_dtype(out["ds"])


def test_maybe_convert_col_to_float32(benchmark, large_panel: pd.DataFrame):
    out = benchmark(maybe_convert_col_to_float32, large_panel, "y")
    assert out["y"].dtype == "float32"


@pytest.mark.parametrize("freq", ["D", "h", "MS"])
def test_get_seasonality(benchmark, freq: str):
    benchmark(get_seasonality, freq)


def test_quantile_converter_from_level(benchmark):
    qc = benchmark(QuantileConverter, [80, 90, 95], None)
    assert qc.quantiles is not None


def test_quantile_converter_level_to_quantiles(
    benchmark,
    large_panel: pd.DataFrame,
):
    """Levels -> quantile columns, as done when users pass `quantiles`."""
    model = "Model"
    df = large_panel.rename(columns={"y": model})
    qc = QuantileConverter(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    for lv in [50, 80]:
        df[f"{model}-lo-{lv}"] = df[model] - lv / 100
        df[f"{model}-hi-{lv}"] = df[model] + lv / 100
    out = benchmark(qc.maybe_convert_level_to_quantiles, df, [model])
    assert f"{model}-q-90" in out.columns


def test_quantile_converter_quantiles_to_level(
    benchmark,
    large_panel: pd.DataFrame,
):
    """Quantile columns -> lo/hi levels, as done when users pass `level`."""
    model = "Model"
    df = large_panel.rename(columns={"y": model})
    qc = QuantileConverter(level=[80, 90])
    for q in qc.quantiles or []:
        df[f"{model}-q-{int(q * 100)}"] = df[model] + q
    out = benchmark(qc.maybe_convert_quantiles_to_level, df, [model])
    assert f"{model}-lo-90" in out.columns
