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
