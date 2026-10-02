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

| Value | Behavior when horizon exog is provided |
|-------|----------------------------------------|
| `"auto"` (default) | Use **native** exog when the checkpoint supports it; otherwise **raise** |
| `False` | Ignore `X_df` / `futr_exog_list`; univariate forecast (also in CV) |

The string `"native"` is accepted as a silent alias for `"auto"`.

When a model does not support native exog and you pass `X_df`, FoundationForecast logs a
warning and raises `ValueError`. For **`FoundationForecast`** with mixed models, set
`exog_strategy=False` on models that do not support native exog so they ignore `X_df`.

## Native exog by checkpoint

FoundationForecast uses **`supports_native_futr_exog()`** on the forecaster instance
(usually a function of **`repo_id`** or API model id). Only these paths accept
`X_df` / CV exog under the default `"auto"` strategy.

| Wrapper | `repo_id` / id pattern | Native futr exog | Notes |
|---------|------------------------|------------------|-------|
| **Chronos** | contains `chronos-2` (case-insensitive), e.g. `amazon/chronos-2` | Yes | Other Chronos checkpoints → error with `X_df` |
| **TimesFM** | contains `3.0`, e.g. `google/timesfm-3.0-pytorch` | Yes | `1.0`, `2.0`, `2.5` → error with `X_df` |
| **TimeGPT** | any Nixtla `model=` id (e.g. `timegpt-1`, `timegpt-2`) | Yes | Hosted API via `X_df` |
| **T0** | `theforecastingcompany/t0-alpha`, `theforecastingcompany/t0-beta` | Yes | `future_covariates` in `tfc-t0` |
| **Tafsut** | any | No | Use `exog_strategy=False` in multi-model runs |
| **Toto** | any | No | Use `exog_strategy=False` in multi-model runs |
| **TiRex** | any | No | Use `exog_strategy=False` in multi-model runs |
| **Moirai** | any | No | Use `exog_strategy=False` in multi-model runs |
| **FlowState** | any | No | Use `exog_strategy=False` in multi-model runs |
| **Sundial** | any | No | Use `exog_strategy=False` in multi-model runs |
| **PatchTST-FM** | any | No | Use `exog_strategy=False` in multi-model runs |
| **TabPFN** | any | No | Use `exog_strategy=False` in multi-model runs |

Local checkpoint paths follow the same rules as Hugging Face `repo_id` strings passed to the
wrapper. For **`cross_validation()`**, pass exog in **`df`** through the holdout window; the
same native rules apply per checkpoint.

## Example

```python
import pandas as pd
from foundationforecast.models import Chronos, Tafsut
from foundationforecast import FoundationForecast

train = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/train.parquet"
)
test = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/test.parquet"
)
# Panel of fev-bench EPF markets (BE, DE, FR, NP, PJM); covariates unified as ex_1, ex_2
exog = ["ex_1", "ex_2"]
X_df = test[["unique_id", "ds", *exog]]

model = Chronos(repo_id="amazon/chronos-2", alias="Chronos-2")
fcst = model.forecast(df=train, h=24, freq="h", X_df=X_df)

# Mixed FoundationForecast: native model + univariate models ignoring X_df
ff = FoundationForecast(
    models=[
        Chronos(repo_id="amazon/chronos-2", alias="Chronos-2"),
        Tafsut(alias="Tafsut", exog_strategy=False),
    ]
)
fcst_panel = ff.forecast(df=train, h=24, freq="h", X_df=X_df)

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
