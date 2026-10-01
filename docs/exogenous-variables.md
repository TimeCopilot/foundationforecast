# Exogenous variables (known-future covariates)

**Exogenous variables (exog)** are **covariates**: extra inputs known over the forecast
horizon. Iteration 1 supports **future-known dynamic** covariates only.

## Data API

- **`df`**: `unique_id`, `ds`, `y`, plus exog columns for **history**.
- **`X_df`**: `unique_id`, `ds`, exog columns for the **next `h` steps** (no `y`).
- Alias: **`futr_df`** = same as **`X_df`**.
- **`futr_exog_list`** (optional on `forecast()` / `cross_validation()`): column names.
  If omitted, infer from `X_df` columns (excluding `unique_id`, `ds`, `y`).

## Model constructor: `exog_strategy`

| Value | Behavior when `X_df` is passed |
|-------|--------------------------------|
| `"auto"` (default) | Native exog if the model supports it; else `XReg()` linear fallback |
| `"native"` | Native only; error if unsupported |
| `XReg(fm_first=True, ...)` | Force FM + linear regressor path |

`XReg.fm_first=True` matches TimesFM `"xreg + timesfm"`; `False` matches `"timesfm + xreg"`.

**TimeGPT:** use `"auto"` or `"native"` only (Nixtla API via `X_df`). `XReg(...)` is rejected.

## Example

```python
import pandas as pd
from foundationforecast.models import Tafsut

train = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/train.parquet"
)
test = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/test.parquet"
)
exog = ["Ampirion Load Forecast", "PV+Wind Forecast"]
X_df = test[["unique_id", "ds", *exog]]

model = Tafsut(exog_strategy="auto")
fcst = model.forecast(df=train, h=24, freq="h", X_df=X_df)
```

See `docs/examples/exogenous-variables.ipynb` for more models.
