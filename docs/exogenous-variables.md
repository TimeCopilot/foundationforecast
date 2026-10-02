# Exogenous variables (known-future covariates)

**Exogenous variables (exog)** are **covariates**: extra inputs known over the forecast
horizon. Iteration 1 supports **future-known dynamic** covariates only.

## Data API

- **`df`**: `unique_id`, `ds`, `y`, plus exog columns. For **history**, exog must cover
  the observed period. For **`cross_validation()`**, also include exog (and `y`) through
  the end of each fold so horizon covariates are read from `df` (no separate horizon frame).
- **`X_df`**: `unique_id`, `ds`, exog columns for the **next `h` steps** (no `y`). Used by
  **`forecast()`** only.
- Alias: **`futr_df`** = same as **`X_df`** on **`forecast()`**.
- **`futr_exog_list`** (optional): column names. On **`forecast()`**, infer from `X_df` if
  omitted. On **`cross_validation()`**, infer from non-target columns in `df` if omitted.

## Model constructor: `exog_strategy`

| Value | Behavior when `X_df` is passed |
|-------|--------------------------------|
| `"auto"` (default) | Native exog if the model supports it; else `XReg()` linear fallback |
| `"native"` | Native only; error if unsupported |
| `XReg(fm_first=True, ...)` | Force FM + linear regressor path |

`XReg.fm_first=True` matches TimesFM `"xreg + timesfm"`; `False` matches `"timesfm + xreg"`.

With `XReg`, pass `level` or `quantiles` as usual: the univariate FM forecast is
computed with intervals first, then the same linear exog adjustment is applied to
the point column and every `{alias}-lo-*`, `{alias}-hi-*`, or `{alias}-q-*` column.

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
# Panel of fev-bench EPF markets (BE, DE, FR, NP, PJM); covariates unified as ex_1, ex_2
exog = ["ex_1", "ex_2"]
X_df = test[["unique_id", "ds", *exog]]

model = Tafsut(exog_strategy="auto")
fcst = model.forecast(df=train, h=24, freq="h", X_df=X_df)

# Cross-validation: one df with exog through the holdout window (no X_df)
panel = pd.concat([train, test], ignore_index=True).sort_values(["unique_id", "ds"])
cv = model.cross_validation(
    df=panel,
    h=24,
    freq="h",
    futr_exog_list=exog,
)
```

See `docs/examples/exogenous-variables.ipynb` for more models.
