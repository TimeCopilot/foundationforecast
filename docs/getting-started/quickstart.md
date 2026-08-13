# Quickstart

## Installation

```bash
pip install foundationforecast
```

Requires Python 3.10 or later. See [Installation](installation.md) for version-specific model notes.

## Hello World

```python
import pandas as pd
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos, Toto

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

ff = FoundationForecast(models=[Chronos(), Toto(context_length=256)])
fcst_df = ff.forecast(df=df, h=12, freq="MS")
cv_df = ff.cross_validation(df=df, h=12, freq="MS")
```

Your DataFrame must include:

- `unique_id` — series identifier (string)
- `ds` — timestamp (datetime)
- `y` — target value (float)

## Plotting (optional)

Install plotting extras:

```bash
pip install "foundationforecast[plot]"
```

```python
from foundationforecast.core.forecaster import Forecaster

Forecaster.plot(df, fcst_df, level=[80])
```

## Next steps

- [Forecaster Quickstart](../examples/forecaster-quickstart.ipynb) — forecast and cross-validate with multiple models
- [Compare Foundation Models](../examples/ts-foundation-models-comparison-quickstart.ipynb) — benchmark models
- [Model Hub](../model-hub.md) — browse all available foundation models
