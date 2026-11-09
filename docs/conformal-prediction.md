# Conformal prediction intervals

FoundationForecast can calibrate **distribution-free prediction intervals** from cross-validation absolute errors on the **point forecast**, instead of using a foundation model’s native quantile head.

## When to use it

- You need **wide levels** (e.g. 95%) from a model whose quantile head stops at fixed knots. Tafsut and Chronos-Bolt predict quantiles 0.1 to 0.9 only, so their native 95% interval is clamped to the 80% one; conformal intervals are not bound by those knots.
- The model produces **only point forecasts** and is not natively probabilistic, e.g. Kumo-Forecast. Without `prediction_intervals`, `level` / `quantiles` raise a `ValueError`; with it you get calibrated intervals.
- You want intervals from **empirical forecast errors** on your own history (per series), rather than the model’s built-in uncertainty.
- You accept the extra compute: about **one panel point forecast per calibration window**, plus the final forecast.

Native quantiles (default when `prediction_intervals` is unset) remain unchanged. **`cross_validation()`** uses conformal intervals too: every window is recalibrated on the data before its cutoff, so backtests show the coverage you would get in production. **`detect_anomalies`** is unchanged and does not read `prediction_intervals`.

## API

Pass [`ConformalIntervals`][foundationforecast.core.conformal.ConformalIntervals] as `prediction_intervals` when creating the model, then pass `level` or `quantiles` to [`forecast()`][foundationforecast.FoundationForecast.forecast]:

```python
import pandas as pd

from foundationforecast import ConformalIntervals, FoundationForecast
from foundationforecast.models import Chronos

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

model = Chronos(
    repo_id="amazon/chronos-bolt-tiny",
    alias="Chronos-Bolt",
    prediction_intervals=ConformalIntervals(n_windows=3),
)

ff = FoundationForecast(models=[model])
fcst = ff.forecast(df=df, h=12, freq="MS", level=[80, 95])
```

### Parameters

| Field | Meaning |
| --- | --- |
| `n_windows` | Rolling backtest windows for residual estimation (minimum **2**). |
| `method` | `conformal_distribution` (default) or `conformal_error`. |
| `step_size` | Gap between calibration cutoffs. `None` (default) uses the forecast `h` (non-overlapping windows). |

The calibration horizon is always the `h` passed to `forecast()` / `cross_validation()`.

### Requirements

- You must pass **`level`** or **`quantiles`** whenever `prediction_intervals` is set.
- Each series needs at least **`h + (n_windows - 1) * step_size + 1`** observations (`step_size` defaults to `h`).

## How it works

Calibration is a panel [`cross_validation()`][foundationforecast.FoundationForecast.cross_validation] of the **point forecast** (one forecast per window for all series). Let \(T\) be the last observed timestamp, \(h\) the horizon, \(W\) = `n_windows`, and \(\Delta\) = `step_size` (default \(h\)).

**1. Backtest the point forecast.** For each window \(w = 1, \dots, W\), the model forecasts \(h\) steps from the cutoff \(t_w = T - h - (W - w)\,\Delta\), using only data up to \(t_w\). The absolute error at horizon step \(j = 1, \dots, h\) is

\[
e_{w,j} = \left| \, y_{t_w + j} - \hat{y}_{t_w + j \mid t_w} \right|.
\]

This gives \(W\) calibration errors per series and horizon step, so later steps, which are usually harder to forecast, get their own (typically larger) errors.

**2. Forecast.** The model produces the point forecast \(\hat{y}_{T+j}\) from the full history, exactly as without conformal intervals.

**3. Build the interval.** For a level \(L\) (in percent), let \(\alpha = 1 - L/100\).

- **`conformal_error`** widens the point forecast symmetrically by the \(L\)% quantile of the calibration errors:

    \[
    \hat{q}_j = \operatorname{Quantile}_{1-\alpha}\big(e_{1,j}, \dots, e_{W,j}\big),
    \qquad
    \big[\, \hat{y}_{T+j} - \hat{q}_j,\; \hat{y}_{T+j} + \hat{q}_j \,\big].
    \]

- **`conformal_distribution`** (default) forms \(2W\) synthetic outcomes around the point forecast and takes their symmetric quantiles:

    \[
    S_j = \big\{ \hat{y}_{T+j} - e_{w,j} \big\}_{w=1}^{W} \cup \big\{ \hat{y}_{T+j} + e_{w,j} \big\}_{w=1}^{W},
    \]

    \[
    \big[\, \operatorname{Quantile}_{\alpha/2}(S_j),\; \operatorname{Quantile}_{1-\alpha/2}(S_j) \,\big].
    \]

The point forecast \(\hat{y}_{T+j}\) is never modified; only the interval columns change. When `quantiles` are requested, each quantile \(q \neq 0.5\) is mapped to the level \(L = |100 - 200\,q|\) and read from the matching lower or upper bound.

**Coverage.** These methods use empirical quantiles without a finite-sample conformal correction, so the requested level is not a coverage guarantee. Even with independent calibration and future errors from the same continuous distribution, two calibration windows give at most two-thirds coverage because the interval radius never exceeds the largest calibration error. Use more calibration windows and measure achieved coverage on held-out backtests, especially when requesting high levels.

See the [conformal prediction example notebook](examples/conformal-prediction.ipynb) on M4 Hourly.
