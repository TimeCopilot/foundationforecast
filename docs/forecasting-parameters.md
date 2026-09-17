# Forecasting Parameters

This section describes how [`Forecaster`][foundationforecast.core.forecaster.Forecaster] methods determine the core forecasting parameters: `freq`, `h`, and seasonality.

You can:

- pass values explicitly as keyword arguments (recommended), or
- let the library infer `freq` from your `ds` column and derive seasonality from it.

### What do these terms mean?

* **`freq`**: the pandas frequency string that describes the spacing of your timestamps (`"H"` for hourly, `"D"` for daily, `"MS"` for monthly-start, etc.).
* **seasonality**: the length of the dominant seasonal cycle in number of `freq` periods (24 for hourly data with a daily cycle, 12 for monthly-start data with a yearly cycle, …). See [`get_seasonality`][foundationforecast.core.forecaster.get_seasonality] for the default mapping.
* **`h` (horizon)**: how many future periods you want to forecast.

!!! tip "Pandas available frequencies"
    See the [pandas offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for valid frequency strings.

## Parameter inference

When you call `forecast()` or `cross_validation()`:

1. **`freq`**: if not provided, [`maybe_infer_freq(df)`][foundationforecast.core.forecaster.maybe_infer_freq] infers it from the most common series in your DataFrame.
2. **`h`**: must be provided explicitly for forecasting and cross-validation.
3. **seasonality**: used internally for anomaly detection when `h` is omitted; defaults to [`get_seasonality(freq)`][foundationforecast.core.forecaster.get_seasonality].

## Explicit parameters

```python
import pandas as pd
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

ff = FoundationForecast(models=[Chronos()])
fcst_df = ff.forecast(df=df, h=12, freq="MS")
```

## Anomaly detection defaults

When calling `detect_anomalies()` without `h`:

* `freq` is inferred from the data if not provided.
* `h` defaults to the seasonal period for the inferred frequency.
* `n_windows` defaults to the maximum number of cross-validation windows supported by the shortest series.

```python
anomalies_df = ff.detect_anomalies(df=df, freq="MS", level=99)
```

## Choosing sensible values

* **`freq`** must match your data's timestamp spacing. Irregular or gapped series may fail inference.
* **`h`** should cover the horizon you care about for evaluation or deployment.
* For **anomaly detection**, use a horizon aligned with the seasonal cycle when possible.


## Probabilistic forecasting (`level` and `quantiles`)

Foundation models support two mutually exclusive ways to request probabilistic
forecasts in `forecast()` and `cross_validation()`:

### `level` — prediction intervals

Pass confidence levels as percentages, e.g. `level=[80, 95]`. Each level `L`
maps to symmetric quantiles `(α/2, 1 − α/2)` with `α = 1 − L/100`. The output
includes `{model}-lo-{L}` and `{model}-hi-{L}` columns.

```python
fcst_df = ff.forecast(df=df, h=12, freq="MS", level=[80, 95])
```

### `quantiles` — direct quantile forecasts

Pass quantile levels in `(0, 1)`, e.g. `quantiles=[0.1, 0.5, 0.9]`. The output
includes `{model}-q-{pct}` columns where `pct = int(100 × quantile)`.

```python
fcst_df = ff.forecast(df=df, h=12, freq="MS", quantiles=[0.1, 0.5, 0.9])
```

### Point forecasts

The point forecast is **always** returned in the `{model}` column, regardless of
whether you use `level` or `quantiles`. You do not need a special level to
request the median.

### Interpolation on fixed-knot models

Some models (TiRex, TimesFM, TabPFN, FlowState, Tafsut, Toto 2.0, …) predict a
fixed set of native quantile **knots** internally. When you request levels or
quantiles that do not exactly match those knots, the library linearly
interpolates between adjacent knots.

### Edge clamping

When a requested quantile falls **outside** the model's native knot range, the
value at the nearest edge knot is returned (same semantics as `numpy.interp`).
Values are not extrapolated beyond the model's trained quantile range.

For example, on a model with native knots `0.1` through `0.9`:

- `level=[95]` maps to quantiles `0.025` and `0.975`, which are clamped to
  `0.1` and `0.9` respectively.
- `quantiles=[0.01]` returns the same forecast as `quantiles=[0.1]`.
