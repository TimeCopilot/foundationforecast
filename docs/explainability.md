# Explainability (Shapley attributions)

`explain()` splits each point forecast into **additive contributions** from the parts of the input the model saw: windows of the target history and known-future exogenous columns. It uses [Shapley values](https://en.wikipedia.org/wiki/Shapley_value) computed by the [`shap`](https://github.com/shap/shap) package, and it works on every model without touching its internals: a part of the input is "switched off" by masking it, and the model is simply run again.

=== "pip"

    ```bash
    pip install "foundationforecast[explain]"
    ```

=== "uv"

    ```bash
    uv add "foundationforecast[explain]"
    ```

!!! warning "Attributions are not causal effects"

    Shapley values describe how **the model's forecast** reacts when part of its input is masked. They do not say that a window or a covariate **causes** the series to move. Masked inputs can also look unlike anything the model saw in training. Missing a feature you need? [Open an issue](https://github.com/TimeCopilot/foundationforecast/issues/new) and tell us what you would like `explain()` to cover.

## When to use it

- You want to know **how much of a forecast comes from last day's values, last week's values, older history, or a specific covariate**, per series and per horizon step.
- You want an explanation that **adds up**: `forecast = base + sum of contributions`, exactly.
- You accept the extra compute: `explain()` re-runs the model on masked inputs (see [Cost](#cost)).

`forecast()` never computes attributions, even when a model has `explanations` set. Attributions only appear when you call `explain()`, so forecasting stays as fast as before.

## API

Every model and [`FoundationForecast`][foundationforecast.FoundationForecast.explain] have an `explain()` method. It takes the same data arguments as `forecast()` (`df`, `h`, `freq`, `X_df`, `futr_exog_list`), without `level` and `quantiles`, plus `ids` and `explanations`.

```python
import pandas as pd

from foundationforecast import FoundationForecast, ShapExplanations
from foundationforecast.models import Chronos

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

model = Chronos(
    repo_id="amazon/chronos-bolt-tiny",
    alias="Chronos-Bolt",
    explanations=ShapExplanations(history=[12, 36]),
)
ff = FoundationForecast(models=[model])

fcst = ff.forecast(df=df, h=3, freq="MS")  # no attribution columns
attr = ff.explain(df=df, h=3, freq="MS")  # forecast + attributions
```

`explanations` on the model constructor only stores **defaults for `explain()`**. Pass `explanations=` to `explain()` to override them for one call; on `FoundationForecast` that override applies to every model:

```python
# This call only: other windows and the cheaper leave-one-out method.
attr = ff.explain(
    df=df,
    h=3,
    freq="MS",
    explanations=ShapExplanations(history=[6], method="loo"),
)
```

## Output

One row per series and future timestamp, with the forecast and one column per component:

| Column | Meaning |
| --- | --- |
| `{alias}` | The point forecast. Equals what `forecast()` returns. |
| `{alias}-shap-base` | Baseline: the mean of the target (see `base_window`). The forecast with every player off. |
| `{alias}-shap-hist-{window}` | Contribution of one history window. |
| `{alias}-shap-{col}` | Contribution of one exogenous column. |
| `{alias}-shap-residual` | Only for `method="loo"`: the part the contributions do not explain. |

For `method="exact"` and `"permutation"`, every row satisfies

\[
\hat{y}_{T+j} = b + \sum_{p} \phi_{p,j},
\]

where \(b\) is the baseline and \(\phi_{p,j}\) the contribution of player \(p\) at horizon step \(j\). For the example above:

```text
 ds          Chronos-Bolt  ...-shap-base  ...-shap-hist-12  ...-shap-hist-12-36  ...-shap-hist-older
 1961-01-01         456.5          280.3             103.5                 32.6                 40.1
```

## Players

**History windows.** `history` cuts the history into windows measured back from the last observation. `history=["1D", "7D"]` on hourly data gives three players: the last day (`hist-1D`), the six days before it (`hist-1D-7D`), and everything older (`hist-older`). Boundaries are either offset strings or a number of steps, and must increase. When `history` is `None` the windows depend on the frequency:

| Frequency | Default boundaries |
| --- | --- |
| sub-daily | `"1D"`, `"7D"` |
| daily | `"7D"`, `"28D"` |
| weekly | 4, 13 steps |
| monthly | 3, 12 steps |
| quarterly | 4, 12 steps |
| yearly | 3 steps |
| business daily | 5, 20 steps |

**Exogenous columns.** When you pass `X_df`, every exogenous column is a player by default (`exog=True`). Set `ShapExplanations(exog=["temp"])` to explain only some columns, or `ShapExplanations(exog=False)` for none. Columns that are not players are always given to the model. Exog players need a model with native known-future exog support (see [Exogenous variables](exogenous-variables.md)); other models raise the same error as `forecast()`.

```python
import pandas as pd

from foundationforecast import ShapExplanations
from foundationforecast.models import Chronos

df = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/train.parquet"
)
X_df = pd.read_parquet(
    "https://timecopilot.s3.amazonaws.com/public/data/electricity_price/futr_exog.parquet"
)

model = Chronos(repo_id="amazon/chronos-2", alias="Chronos-2")
attr = model.explain(
    df=df,
    h=24,
    X_df=X_df,
    ids=["DE"],
    explanations=ShapExplanations(history=["1D", "7D"]),
)
```

The result has `Chronos-2-shap-hist-1D`, `-hist-1D-7D`, `-hist-older`, `-ex_1`, and `-ex_2`.

!!! warning "History windows larger than the model's context"

    A history window is only useful if the model can read it. **Expected behavior:**

    - A window that has no observations (a boundary past the start of a series), or that lies entirely beyond the model's `context_length`, cannot change what the model reads. It gets **zero attribution** and `explain()` raises a `UserWarning` naming the window. It does not cost extra model runs.
    - A window the model reads only in part stays a player. With `history=[1, 30]` and `context_length=20`, the window `1-30` is explained from the 19 steps the model sees.
    - The check is per series, so in a panel with series of different lengths a window can be empty for some series and not for others.

    **Limit:** only a `context_length` that the model wrapper exposes is detected. A checkpoint with a hidden internal cap is not, and a window beyond that cap can still pick up credit that is not real: a coalition in which only that window is on masks everything the model actually reads. Set `context_length` on the model, or keep `history` boundaries within what the checkpoint reads.

## How a player is switched off

Each coalition of players that stay on becomes a masked copy of your data, and the model forecasts it.

- **Exogenous column off:** it is removed from the model inputs.
- **History window off:** the oldest windows are cut from the start of the context. A window that is off but newer than a window that is on is masked in place, because the forecast must still start at the last observed timestamp. Masking in place uses `mask`:
    - `"nan"` sets the target values to NaN. Models that treat NaN as missing (Chronos, Tafsut, Timestral, TiRex, and Moirai 2) use this under `mask="auto"`.
    - `"fill"` replaces the values with the series baseline. Every other model uses this under `mask="auto"`, so no model ever receives NaN unless it handles it.
- **Every history window off:** the forecast is the baseline, a constant per series, and the model is not called.

No background dataset is sampled: a player is either on or off.

## Methods

| `method` | What it does | Model runs per series |
| --- | --- | --- |
| `"auto"` (default) | `"exact"` when `2**n` is at most the permutation budget, otherwise `"permutation"`. | |
| `"exact"` | Evaluates every coalition. | up to `2**n` |
| `"permutation"` | Samples `n_samples` permutations. | up to `2 * n * n_samples + 1` |
| `"loo"` | Removes one player at a time. Cheap, but not additive, so a `-shap-residual` column is added. | `n + 1` |

Here \(n\) is the number of players. With the default `n_samples=10`, `"auto"` picks `"exact"` up to 7 players and `"permutation"` beyond that. Coalitions are cached, so repeated permutations do not repeat model runs. `"exact"` and `"permutation"` need `foundationforecast[explain]`; `"loo"` does not.

## Cost

`explain()` re-runs the model once per coalition, for every explained series. Nothing is capped. Instead, a `UserWarning` is raised when the estimated number of series forecasts exceeds `warn_model_calls` (default 2,000), and the run continues. Ways to reduce cost:

- `ids=[...]` explains only some series (the same argument name as in `plot`).
- Fewer or coarser `history` windows, or `exog=False`.
- `method="permutation"` with a small `n_samples`, or `method="loo"`.
- `max_batch_series` bounds how many series are sent to the model at once, to limit memory.

Only an unusable request is refused: `method="exact"` with more than 20 players raises `ValueError`, because `2**n` coalitions would not fit in memory (a warning is raised above 12).

## Things to keep in mind

- Attributions describe **this model's behavior on masked inputs**, not causal effects (see the warning above). Masked inputs (a gap, a shortened context) can be unlike anything the model saw in training.
- The baseline is a constant (the series mean), not a model run, so large `base` values are normal. Set `base_window` to use only recent history.
- A shortened context must still be long enough for the model. If masking leaves too little history for a checkpoint, the model raises its usual error; use fewer or longer windows.
- Models that sample (for example, Moirai 1.x) add noise to the differences between coalitions. Contributions are still additive for the runs that were evaluated, but small contributions may be noise.
- `explain()` explains the point forecast. It does not take `level` or `quantiles`.

## Example notebook

See the [explainability example notebook](examples/explainability.ipynb) for a single-model walk-through with a planted-signal check, several models at once, `nan` vs `fill` masking, and the context limit above.

## Settings

See [`ShapExplanations`][foundationforecast.core.shap.ShapExplanations] for every option.
