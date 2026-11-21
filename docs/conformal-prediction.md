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
| `method` | `conformal_distribution` (default), `conformal_error`, `conformal_error_finite_sample`, or a method object: [`PID(...)`][foundationforecast.core.conformal_pid.PID] or [`ACI(...)`][foundationforecast.core.conformal_aci.ACI]. |
| `step_size` | Gap between calibration cutoffs. `None` (default) uses the forecast `h` (non-overlapping windows). |

The calibration horizon is always the `h` passed to `forecast()` / `cross_validation()`.

### Requirements

- Pass **`level`** or **`quantiles`** to produce intervals. Without either, only the point forecast is returned.
- Each series needs at least **`h + (n_windows - 1) * step_size + 1`** observations (`step_size` defaults to `h`).
- `conformal_error_finite_sample` and `ACI` require finite calibration errors; `NaN` or infinite errors raise `ValueError`.

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

- **`conformal_error_finite_sample`** applies the finite-sample correction to the absolute-error method. Sort the \(W\) errors at each series and horizon step, then select the **\(k\)-th smallest** (one-based), where

    \[
    k = \left\lceil (W + 1)\,L/100 \right\rceil.
    \]

    This is the inverse empirical-CDF quantile at the corrected level \(k/W\), with no interpolation. The interval is \([\hat y_{T+j}-e_{(k),j},\;\hat y_{T+j}+e_{(k),j}]\). If \(k>W\), the radius is \(+\infty\), giving bounds **`-inf` and `inf`**. The method never clips the corrected level to 1. Each series and horizon step has \(W\) calibration errors; the \(2W\) synthetic outcomes in `conformal_distribution` are not additional independent calibration samples.

The point forecast \(\hat{y}_{T+j}\) is never modified; only the interval columns change. When `quantiles` are requested, each quantile \(q \neq 0.5\) is mapped to the level \(L = |100 - 200\,q|\) and read from the matching lower or upper bound.

### Finite-sample correction and coverage

Select the corrected method explicitly; the existing methods and default are unchanged:

```python
corrected_intervals = ConformalIntervals(
    n_windows=19,
    method="conformal_error_finite_sample",
)
```

Pass this configuration as a model's `prediction_intervals`. At 95%, **19 windows** are the minimum for a finite radius; 90% needs 9, and 80% needs 4. For example, two calibration windows produce infinite 80% and 95% bounds. At forecast time, a **`UserWarning`** lists the unsupported levels and the minimum number of calibration windows each needs, for example: `level 95 needs at least 19 calibration windows, got 4; bounds are infinite`. The bounds remain infinite. Increase `n_windows` to obtain finite intervals, subject to the history-length requirement above. Check for infinite bounds before plotting or computing metrics that require finite values.

**Compute and history cost.** With `n_windows=19`, each `forecast()` runs about **20 panel point forecasts per model**: 19 calibration forecasts plus the final forecast. Each series needs at least **`h + 18 * step_size + 1`** historical observations, or **457 observations** when `h=24` and `step_size=24`. A rolling-origin evaluation repeats this calibration at every evaluation cutoff.

The correction gives the usual **marginal coverage** lower bound when the calibration and future absolute errors are exchangeable. It does not guarantee simultaneous coverage of the entire forecast horizon. Rolling-origin residuals use different training histories, can be dependent, and may drift; even non-overlapping windows do not establish exchangeability. The correction therefore does **not** by itself guarantee nominal coverage on time series. See [Angelopoulos and Bates, Section 1.1](https://arxiv.org/abs/2107.07511).

The original `conformal_error` and `conformal_distribution` use interpolated empirical quantiles without this correction. Even with independent calibration and future errors from the same continuous distribution, two calibration windows give at most two-thirds coverage because their radius never exceeds the largest calibration error. Use more calibration windows and measure achieved coverage on held-out backtests, especially when requesting high levels.

The [conformal prediction example notebook](examples/conformal-prediction.ipynb) includes an M4 Hourly rolling-origin check comparing corrected and uncorrected intervals, with empirical coverage, mean width, and the fraction of finite intervals. Calibration uses only observations available before each evaluation cutoff. These results measure that dataset and configuration, rather than proving a coverage guarantee. For adaptive replay see PID and ACI below.

## Conformal PID control under distribution shift

The methods above treat the calibration windows as one pool, so they react slowly when the error scale changes (a volatility jump, a new regime). `PID` replays the calibration windows **in time order** and tracks the error quantile online, following [Conformal PID Control for Time Series Prediction](https://arxiv.org/abs/2307.16895) (Angelopoulos, Candes, Tibshirani, 2023). Pass a method object instead of a string:

```python
from foundationforecast import PID, ConformalIntervals

intervals = ConformalIntervals(n_windows=20, method=PID(lr=0.1))
```

Per horizon step, and separately for the lower and upper tail (each targets \(\alpha/2\) misses), the threshold \(\hat q\) follows

\[
\hat q_{k+1} = \underbrace{\hat q^{P}_{k} + \eta_k\,(\mathrm{err}_k - \alpha/2)}_{\text{P: quantile tracking}}
\;+\; \underbrace{K_I \tan\!\Big(\tfrac{\log n}{C_{\mathrm{sat}}\,n} \textstyle\sum_{i\le k}(\mathrm{err}_i - \alpha/2)\Big)}_{\text{I: error integration}},
\]

where \(\mathrm{err}_k = 1\) if the score of window \(k\) exceeded the threshold that was issued for it, \(\eta_k = \texttt{lr} \times\) (range of the errors seen so far) puts the step on the score scale, and \(\tan\) saturates to \(\pm\infty\) beyond \(\pm\pi/2\). The P term moves the threshold up after a miss and down after a cover. The I term corrects a persistent coverage bias that P alone reacts to slowly. The scores are the signed errors \(y - \hat y\) (lower tail uses \(-e\)), so intervals can be asymmetric; `symmetric=True` tracks \(|e|\) instead. The final threshold is floored at zero (the interval always contains the point forecast) and sorted across levels. The scorecasting (D) module of the paper is not implemented.

**Delayed feedback.** The error at horizon step \(j\) of a window is only observable \(j\) observations after that window's cutoff. With `step_size < h` the replay only uses errors that had been revealed by each cutoff, so the replay never sees the future. The forecast itself sits \(h\) steps after the last window and sees every window.

| Field | Meaning |
| --- | --- |
| `lr` | Base learning rate of quantile tracking (default `0.1`), scaled by the range of the errors. |
| `symmetric` | Track \(|e|\) with one threshold (`True`) or each tail separately (`False`, default). |
| `integrate` | Include the error-integration term (default `True`). |
| `KI`, `Csat`, `Tg`, `delta` | Integrator scale and saturation. Defaults follow the paper: `KI` is the largest absolute calibration error of the series, `Csat` is derived from `Tg` (default `n_windows`) and `delta` (default `0.01`). |
| `ncal`, `rolling` | Windows used to warm start (default `n_windows // 2`); with `rolling=True`, also the trailing window of the learning-rate range. |
| `init` | `"quantile"` (default) starts from the empirical quantile of the first `ncal` windows and replays the rest. `"zero"` starts from zero and replays every window, as in the reference formulation, which needs many windows to warm up. |

When the integrator saturates (the miss rate is far from the target, typically at high levels with few windows) the bound is infinite and `forecast()` emits a `UserWarning`, as with the finite-sample correction. Add windows, raise `Csat`/`Tg`, or set `integrate=False`.

The default integrator needs `n_windows >= 3` (with `Tg=n_windows` two windows give a non-positive `Csat`); `ConformalIntervals` raises at construction otherwise. Use more windows, pass `Tg`/`Csat`, or set `integrate=False`. With `init="quantile"` and `step_size < h`, warm-start windows whose score is not yet revealed at the first origin are not replayed later, as in R's `conformalForecast::pid`.

**Verification.** The implementation is cross-checked against R's [`conformalForecast::pid`](https://github.com/xqnwang/conformalForecast) (Wang, Hyndman, Girolimetto): `tests/core/test_conformal_pid_r.py` reproduces every interval R emits (asymmetric/symmetric, expanding/rolling, with/without integrator, all horizons) to `1e-9` from a stored fixture, regenerated with `tests/core/fixtures/pid_r/generate.py`. Hand-computed golden values, an independent calendar-time oracle and an out-of-sample coverage test are covered in `tests/core/test_conformal_pid.py`.

### Illustration

For a runnable walkthrough on real data, GOOGL daily prices with changing volatility, comparing the model's own intervals, `conformal_error` and `PID`, see the [conformal prediction under distribution shift notebook](examples/conformal-distribution-shift.ipynb).

Chronos-2 rolling-origin forecasts on M4 Hourly (414 series, \(h=24\), non-overlapping windows). The calibrators see only windows before each evaluated one; coverage is the mean empirical coverage and MSIS the interval score scaled by seasonal-naive MAE (lower is better). "Shift" multiplies the forecast errors of the evaluated windows by 2.5, emulating a volatility regime change the calibration history has not seen. These are one dataset and configuration, not a coverage guarantee.

| Method | Coverage 80 / 95 | MSIS 80 / 95 | Shift: coverage 80 / 95 | Shift: MSIS 80 / 95 |
| --- | --- | --- | --- | --- |
| native quantiles | 0.754 / 0.960 | 3.42 / 5.59 | 0.356 / 0.654 | 11.9 / 24.2 |
| `conformal_error`, 20 windows | 0.770 / 0.906 | 3.70 / 6.28 | 0.491 / 0.697 | 11.5 / 25.5 |
| `PID()` | 0.798 / 0.962 | 4.24 / 6.36 | **0.657 / 0.835** | **10.3 / 18.2** |
| `PID(integrate=False)` | 0.790 / 0.922 | 3.59 / 6.09 | 0.530 / 0.677 | 10.7 / 26.0 |

With stable errors PID reaches nominal coverage and is comparable to a 20-window split conformal at similar cost; `integrate=False` gives tighter intervals there. Under the shift the integrator matters: the default recovers most of the lost coverage while the pooled and the P-only variants stay far below nominal. On a spiky hourly price series (electricity) the same PID setup over-covers and is wider than split conformal on stable windows, so check the achieved coverage on your own backtests.

## Adaptive conformal inference via replay

`ACI` implements the adaptive miscoverage update of [Gibbs and Candes (2021)](https://arxiv.org/abs/2106.00170), using the same stateless `ConformalMethod` interface as PID:

```python
from foundationforecast import ACI, ConformalIntervals

aci_intervals = ConformalIntervals(n_windows=20, method=ACI(gamma=0.005))
```

Pass `aci_intervals` as a model's `prediction_intervals` and request `level` or `quantiles`. ACI uses symmetric absolute-error intervals. PID tracks a threshold on the error scale; ACI instead changes the probability used to select an error quantile. For a target level `L`, it initializes `target_alpha = 1 - L/100` and updates a separate adaptive alpha per series, horizon step and level:

\[
\alpha_{k+1} = \alpha_k + \gamma\big(\alpha_{\mathrm{target}} - \mathrm{miss}_k\big).
\]

A miss decreases alpha and requests a higher quantile; a cover increases alpha. Before each replay decision, the threshold is the **inverse empirical-CDF quantile** of previously revealed absolute errors: the `ceil(n_seen * (1 - alpha))`-th smallest score, without interpolation. This is the empirical quantile used for the ACI controller, **not** the finite-sample-corrected rank of `conformal_error_finite_sample`; that method's minimum-window counts do not apply to ACI.

| Field | Meaning |
| --- | --- |
| `gamma` | Adaptation rate in `(0, 1]`, default `0.01`; the example uses `0.005`. Larger values make alpha respond more quickly and may produce extreme intervals. |
| `ncal` | Initial windows reserved for warming up the score distribution, default `max(1, n_windows // 2)`; must be a positive integer smaller than `n_windows`. |
| `rolling` | Use the last `ncal` revealed scores for the CDF (`True`), or every revealed score (`False`, default). |

**Time ordering and delayed feedback.** At horizon step `j`, the score of window `k` is available only at `origin_k + j`. Each replay decision uses only scores available by that cutoff, including when `step_size < h`. Warm-up windows never update alpha; their scores enter the CDF when revealed. Each remaining window updates alpha exactly once, judged against the threshold originally issued for that window, rather than a threshold recomputed after its actual was observed.

**State and coverage.** Every `forecast()` reconstructs this process from its calibration windows and discards the adaptive state afterwards. The same configuration can be reused across calls and models; there is no persistent online session or `update()` API. Levels are simulated independently, then final radii are widened as needed to keep output intervals nested. This resets the controller and modifies the returned sets, so the API **does not inherit the continuous online algorithm's long-run coverage guarantee**. Neither nominal coverage at every cutoff nor simultaneous coverage of the full horizon is promised. In `cross_validation()`, every evaluation cutoff gets a fresh replay using only its training history.

**Extreme intervals are explicit.** Adaptive alpha is not clipped. At `alpha <= 0`, the interval is unbounded (`-inf`, `inf`); at `alpha >= 1`, it is empty (`+inf`, `-inf`). A decision with no revealed scores also starts unbounded. ACI emits a `UserWarning` naming the affected output levels and the replay settings whenever returned intervals are unbounded or empty. Check for these sets before plotting or scoring; use more calibration windows or review `gamma` and `ncal`. Small positive alpha can also select the largest observed error, without giving a finite-sample coverage guarantee.

**Compute and history cost.** Like the other methods, `n_windows=20` requires about 21 panel point forecasts per model and at least `h + 19 * step_size + 1` historical observations per series (481 with `h=step_size=24`). Replay adds quantile calculations but no extra foundation-model forecasts. Rolling-origin evaluation repeats calibration at every cutoff.

The [ACI replay notebook](examples/conformal-aci.ipynb) starts with a finite 24-hour forecast and plot on one M4 Hourly series, then compares ACI, PID and static finite-sample-corrected intervals on held-out rolling origins. It reports coverage, width, finite fraction and empty fraction together, so trivial coverage from infinite intervals is visible. The learning rate and window counts are illustrative settings to validate on your own series, not universal defaults for good coverage.
