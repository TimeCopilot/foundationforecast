# Prophet adapter

[`FoundationProphet`][foundationforecast.prophet.FoundationProphet] lets you use any foundation model inside an existing [Prophet](https://github.com/facebook/prophet) pipeline. It is a `prophet.Prophet` subclass, so `fit`, `make_future_dataframe`, `predict` and `plot` keep working, but the forecasts come from the model you pass.

![Swap Prophet for a foundation model: the same Prophet pipeline with Prophet() and with FoundationProphet(model=...), then switching between Timestral, Chronos-2 and TimesFM](https://timecopilot.s3.amazonaws.com/public/data/images/prophet_adapter_before_after.gif)

## Installation

Prophet is an optional dependency:

```bash
uv add "foundationforecast[prophet]"
# or
pip install "foundationforecast[prophet]"
```

## Swap Prophet for a foundation model

```python
import pandas as pd

from foundationforecast.models import Chronos
from foundationforecast.prophet import FoundationProphet

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

model = Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt")
m = FoundationProphet(model=model)  # instead of Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=12, freq="MS", include_history=False)
forecast = m.predict(future)
print(forecast.head())
```

`df` is a regular Prophet dataframe with `ds` and `y` (a single series). `predict` returns Prophet's usual `ds`, `yhat`, `yhat_lower` and `yhat_upper` columns, one row per row of `future`.

`interval_width` works as in Prophet and is converted to the model's `level` (`0.8` becomes `80`). It must be a whole percent:

```python
m = FoundationProphet(model=model, interval_width=0.95)
m.fit(df)
forecast = m.predict(m.make_future_dataframe(periods=12, freq="MS", include_history=False))
```

## Historical rows

Foundation models are zero-shot, so there are no fitted values for the historical dates. `make_future_dataframe` includes them by default (`include_history=True`), and `FoundationProphet` has two ways to handle them, controlled by `insample`:

| `insample` | Historical rows in `predict` |
| --- | --- |
| `"cv"` (default) | Rolling-origin backtest forecasts from `model.cross_validation`. The model runs once per window. |
| `"none"` | `NaN`. No extra model runs. |

The first time the backtest runs, a message is logged with the number of windows and how to skip them. Tune it with `insample_h` (horizon per window, by default the number of future periods) and `insample_n_windows` (maximum number of windows counted back from the end of the history, 10 by default). Rows outside the windows are `NaN`. Passing `include_history=False` to `make_future_dataframe` skips the backtest.

```python
m = FoundationProphet(model=model, insample="cv", insample_n_windows=3)
m.fit(df)
future = m.make_future_dataframe(periods=12, freq="MS")  # history + 12 months
forecast = m.predict(future)
assert forecast["yhat"].notna().sum() == 3 * 12 + 12
```

## Regressors

Columns added with `add_regressor` are passed to the model as known-future exogenous variables, so the model must support them (see [Exogenous variables](exogenous-variables.md)). The `future` dataframe needs a value for every future period. This example forecasts the next 24 hours of an electricity price using two covariates that are known in advance:

```python
base = "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/"
history = pd.read_parquet(base + "train.parquet").query("unique_id == 'BE'")
future = pd.read_parquet(base + "futr_exog.parquet").query("unique_id == 'BE'")
# columns: ds, y, ex_1, ex_2 (history) and ds, ex_1, ex_2 (next 24 hours)

m = FoundationProphet(
    model=Chronos(repo_id="amazon/chronos-2", alias="Chronos-2"),
    insample="none",
)
m.add_regressor("ex_1")
m.add_regressor("ex_2")
m.fit(history.tail(24 * 28).drop(columns="unique_id"))

forecast = m.predict(future.drop(columns="unique_id"))
```

## Prophet's diagnostics

`prophet.diagnostics.cross_validation` and `performance_metrics` work with `FoundationProphet`: every clone reuses the model you passed.

## What is different from Prophet

- `growth`, `holidays`, seasonalities, changepoints and the other Prophet arguments are accepted so existing code keeps running, but they do not affect a foundation model, and a `UserWarning` tells you so (`add_seasonality` and `add_country_holidays` warn too).
- `plot_components` and the posterior sampling methods raise `NotImplementedError`, as there are no trend or seasonality components.
- The history must be on a regular grid with no gaps. Rows with a null `y` are dropped, so an interior one counts as a gap and `fit` raises; fill or remove them first. The frequency is inferred from `ds`, and dates after the history must continue it, as `make_future_dataframe` does.
- Any model works, as long as it is a foundationforecast model. A future ensemble model can be passed the same way.
