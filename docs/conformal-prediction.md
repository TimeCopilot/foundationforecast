# Conformal prediction intervals

FoundationForecast can calibrate **distribution-free prediction intervals** from cross-validation absolute errors on the **point forecast**, instead of using a foundation model’s native quantile head.

## When to use it

- You want intervals from **empirical forecast errors** on your history (per series).
- A checkpoint has **no native quantiles**, or you prefer **CV-calibrated** bands over the model’s built-in uncertainty.
- You accept the extra compute: roughly **one point forecast per calibration window per series**, plus the final forecast.

Native quantiles (default when `prediction_intervals` is unset) remain unchanged. **`detect_anomalies`** is unchanged and does not read `prediction_intervals`.

## API

Set [`ConformalIntervals`][foundationforecast.core.conformal.ConformalIntervals] on the model, then pass `level` or `quantiles` to [`forecast()`][foundationforecast.FoundationForecast.forecast]:

```python
import pandas as pd

from foundationforecast import ConformalIntervals, FoundationForecast
from foundationforecast.models import Chronos

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

model = Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt")
model.prediction_intervals = ConformalIntervals(
    n_windows=3,
    method="conformal_distribution",
)

ff = FoundationForecast(models=[model])
fcst = ff.forecast(df=df, h=12, freq="MS", level=[80, 90])
```

### Parameters

| Field | Meaning |
| --- | --- |
| `n_windows` | Rolling backtest windows for residual estimation (minimum **2**). |
| `h` | Set automatically from `forecast(h=...)` on each call. |
| `method` | `conformal_distribution` (default) or `conformal_error`. |

Intervals are computed **per `unique_id`**: residuals use the same scale as `y` for that series.

### Requirements

- You must pass **`level`** or **`quantiles`** whenever `prediction_intervals` is set.
- Each series needs enough history: at least **`n_windows * h + 1`** observations (and at least **`2 * h + 1`** when windows are capped by length).

## Methods

- **`conformal_distribution`**: forms synthetic paths from point forecast ± calibration errors and takes symmetric quantiles (wider bands when errors are heavy-tailed).
- **`conformal_error`**: uses quantiles of absolute errors per horizon step around the point forecast.

See the [conformal prediction example notebook](examples/conformal-prediction.ipynb) on electricity prices.
