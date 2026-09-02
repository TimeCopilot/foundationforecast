# foundationforecast

Foundation time series forecasting models, extracted from [TimeCopilot](https://github.com/TimeCopilot/timecopilot).

Run state-of-the-art pretrained models (Chronos, Moirai, TimesFM, Toto, TiRex, TimeGPT, and more) through a single unified API.

## Installation

```bash
pip install foundationforecast
```

Requires Python 3.10+. Some models have additional version requirements — see the [Model Hub](docs/model-hub.md).

Optional plotting support:

```bash
pip install "foundationforecast[plot]"
```

## Quick example

```python
import pandas as pd
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos, Toto

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

ff = FoundationForecast(models=[Chronos(), Toto(context_length=256)])
fcst = ff.forecast(df, h=12, freq="MS")
cv = ff.cross_validation(df, h=12, freq="MS")
```

## Supported models

Chronos, FlowState, Moirai, PatchTST-FM, Sundial, T0, TabPFN, TiRex, TimeGPT, TimesFM, Toto

## Documentation

Build and serve docs locally:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

See [Getting Started](docs/getting-started/quickstart.md) and [Examples](docs/examples/index.md).

## Development

```bash
uv sync --group dev --group docs
pre-commit install --install-hooks
uv run pytest
```

## License

MIT
