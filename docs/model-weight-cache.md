# Model Weight Cache

Foundation models load large checkpoint files into GPU or CPU memory. By default, foundationforecast **reuses loaded weights** across repeated `forecast()` calls instead of loading and unloading on every request.

This is especially useful in long-lived workers (for example Modal GPU containers) where the same model serves many requests.

## Defaults

| Setting | Default | Scope |
| ------- | ------- | ----- |
| `reuse_loaded_model` | `True` | Per forecaster instance |
| `max_cached_models` | `1` | Process-wide LRU cache |

When the cache is full, the least-recently-used model is evicted and its memory is released.

## Per-model opt-out

Every forecaster accepts `reuse_loaded_model` at construction time:

```python
from foundationforecast.models import Chronos, TimesFM, TiRex

chronos = Chronos(repo_id="amazon/chronos-bolt-mini", reuse_loaded_model=True)
timesfm = TimesFM(reuse_loaded_model=False)  # load fresh weights every forecast()
```

Set `reuse_loaded_model=False` when you need strict isolation between calls (for example memory debugging or tests).

## Process-wide cache size

Configure how many loaded models to keep in memory at once:

```python
from foundationforecast import set_max_cached_models

set_max_cached_models(1)   # default; best for single-GPU workers
set_max_cached_models(0)   # disable caching entirely (always load + release)
set_max_cached_models(2)   # keep two models warm; useful when rotating models
```

Call `set_max_cached_models()` once at worker startup (before the first forecast).

## Manual cache release

Each forecaster exposes `clear_model_cache()` to drop its cached weights:

```python
from foundationforecast.models import Chronos

model = Chronos(repo_id="amazon/chronos-bolt-mini")
# ... run model.forecast(df, h=12, freq="D") ...
model.clear_model_cache()
```

## Multi-model ensembles

`FoundationForecast(clean_cache=True)` calls `clear_model_cache()` on every model after each one runs. Use this when forecasting with several large models in one pass and you need to limit peak GPU memory:

```python
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos, TimesFM

FoundationForecast(
    models=[
        Chronos(alias="Chronos"),
        TimesFM(alias="TimesFM"),
    ],
    clean_cache=True,
)
```

## Which models use the cache?

All forecasters that load local weights (Hugging Face checkpoints, GluonTS modules, TabPFN local mode, etc.) go through the shared cache on [`Forecaster`][foundationforecast.core.forecaster.Forecaster].

Remote API models such as **TimeGPT** accept `reuse_loaded_model` for API consistency but do not load local weights.

**Chronos finetuning:** when `finetuning_config` is set, weights are not cached because the model is mutated during `forecast()`.
