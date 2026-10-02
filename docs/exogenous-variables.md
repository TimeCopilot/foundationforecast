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
- **`futr_exog_list`** (optional): exog column names only (not `unique_id`, `ds`, or `y`).
  On **`forecast()`**, infer from `X_df` if omitted. On **`cross_validation()`**, infer
  from non-target columns in `df` if omitted.

## Model constructor: `exog_strategy`

| Value | Behavior when `X_df` is passed |
|-------|--------------------------------|
| `"auto"` (default) | Native exog if the model supports it; else `XReg()` linear fallback |
| `"native"` | Native only; error if unsupported |
| `False` | Ignore `X_df` / `futr_exog_list`; univariate forecast (also in CV) |
| `XReg(fm_first=True, ...)` | Force FM + linear regressor path |

`XReg.fm_first=True` matches TimesFM `"xreg + timesfm"`; `False` matches `"timesfm + xreg"`.

With `XReg`, pass `level` or `quantiles` as usual: the univariate FM forecast is
computed with intervals first, then the same linear exog adjustment is applied to
the point column and every `{alias}-lo-*`, `{alias}-hi-*`, or `{alias}-q-*` column.

**TimeGPT:** use `"auto"` or `"native"` only (Nixtla API via `X_df`). `XReg(...)` is rejected.

## Native vs XReg by checkpoint

FoundationForecast decides native vs fallback from **`supports_native_futr_exog()`** on the
forecaster instance (usually a function of **`repo_id`** or API model id). With
`exog_strategy="auto"`, native runs when supported; otherwise **`XReg()`** (linear exog on
top of the univariate FM forecast) is used.

| Wrapper | `repo_id` / id pattern | Native futr exog | Notes |
|---------|------------------------|------------------|-------|
| **Chronos** | contains `chronos-2` (case-insensitive), e.g. `amazon/chronos-2` | Yes | T5 / Bolt / other Chronos checkpoints → **XReg** |
| **TimesFM** | contains `3.0`, e.g. `google/timesfm-3.0-pytorch` | Yes | `1.0`, `2.0`, `2.5` PyTorch checkpoints → **XReg** |
| **TimeGPT** | any Nixtla `model=` id (e.g. `timegpt-1`, `timegpt-2`) | Yes | Hosted API; **`XReg(...)` rejected** |
| **T0** | `theforecastingcompany/t0-alpha`, `theforecastingcompany/t0-beta` | Yes | `future_covariates` in `tfc-t0` |
| **Tafsut** | any | No | **XReg** only |
| **Toto** | any | No | **XReg** only |
| **TiRex** | any | No | **XReg** only |
| **Moirai** | any | No | **XReg** only |
| **FlowState** | any | No | **XReg** only |
| **Sundial** | any | No | **XReg** only |
| **PatchTST-FM** | any | No | **XReg** only |
| **TabPFN** | any | No | **XReg** only |

Local checkpoint paths follow the same rules as Hugging Face `repo_id` strings passed to the
wrapper. For **`cross_validation()`**, pass exog in **`df`** through the holdout window; native
vs XReg follows the same table per series checkpoint.

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
