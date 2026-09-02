<div align="center">
  <img src="docs/assets/logo-dark.svg#gh-dark-mode-only" alt="FoundationForecast" width="600">
  <img src="docs/assets/logo-light.svg#gh-light-mode-only" alt="FoundationForecast" width="600">
</div>
<div align="center">
  <em>One API for every time series foundation model · Forecast · Cross-validate · Detect anomalies</em>
</div>
<div align="center">
  <a href="https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml"><img src="https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/foundationforecast"><img src="https://img.shields.io/pypi/v/foundationforecast.svg" alt="PyPI"></a>
  <a href="https://github.com/TimeCopilot/foundationforecast"><img src="https://img.shields.io/pypi/pyversions/foundationforecast.svg" alt="versions"></a>
  <a href="https://github.com/TimeCopilot/foundationforecast/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TimeCopilot/foundationforecast" alt="license"></a>
  <a href="https://timecopilot.dev/foundationforecast/"><img src="https://img.shields.io/badge/docs-online-blue" alt="docs"></a>
</div>

---

**FoundationForecast** is a focused library for running state-of-the-art **time series foundation models** through a single unified Python API. It is the foundation-model layer extracted from [TimeCopilot](https://github.com/TimeCopilot/timecopilot) — swap Chronos for Moirai, TimesFM for Toto, or run them side by side with the same three lines of code.

Developed with 💙 by the [TimeCopilot](https://timecopilot.dev/) team.

---

## One API for a fragmented landscape

The forecasting community has seen an explosion of time series foundation models (TSFMs): Chronos, Moirai, TimesFM, TimeGPT, Toto, TiRex, and many more. Each brings different inductive biases and strong benchmark results — but also its own API, dependencies, data conventions, and learning curve.

That fragmentation makes it hard to compare models fairly, let alone use them in production. As we discuss in the [TimeCopilot paper](https://arxiv.org/abs/2509.00616), every lab ships a different interface, training pipeline, and evaluation setup. **FoundationForecast** removes that friction: one `FoundationForecast` class, one DataFrame format, and the same methods — `forecast`, `cross_validation`, and `detect_anomalies` — across every supported model.

---

## Quick example

```python
import pandas as pd
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos, Toto

df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
)

ff = FoundationForecast(models=[Chronos(), Toto(context_length=256)])

fcst = ff.forecast(df, h=12, freq="MS", level=[90])
cv = ff.cross_validation(df, h=12, freq="MS", level=[90])
anomalies = ff.detect_anomalies(df, freq="MS", level=99)
```

Your DataFrame needs three columns: `unique_id`, `ds`, and `y`. Dates in `ds` are parsed automatically — no need to pass `parse_dates`.

---

## Supported models

Every model supports **forecast**, **cross-validation**, and **anomaly detection** through the same API. **Intervals** means prediction intervals via `level` or quantile forecasts. **Finetune** marks models that can adapt to your data at inference time.

| | Model | Forecast | CV | Anomalies | Intervals | Finetune | Notes |
|:-:|---|:-:|:-:|:-:|:-:|:-:|---|
| <img src="docs/assets/logos/amazon.png" width="20" alt=""> | [Chronos](https://arxiv.org/abs/2403.07815) | ✓ | ✓ | ✓ | ✓ | ✓ | Finetune on Chronos 2 |
| <img src="docs/assets/logos/ibm.png" width="20" alt=""> | [FlowState](https://arxiv.org/abs/2508.05287) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/salesforce.png" width="20" alt=""> | [Moirai](https://arxiv.org/abs/2402.02592) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/ibm.png" width="20" alt=""> | [PatchTST-FM](https://arxiv.org/abs/2602.06909) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/thuml.png" width="20" alt=""> | [Sundial](https://arxiv.org/abs/2502.00816) | ✓ | ✓ | ✓ | ✓ | — | Python 3.10–3.12 |
| <img src="docs/assets/logos/tfc.png" width="20" alt=""> | [T0](https://huggingface.co/theforecastingcompany/t0-alpha) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/priorlabs.png" width="20" alt=""> | [TabPFN](https://arxiv.org/abs/2501.02945) | ✓ | ✓ | ✓ | ✓ | — | Python 3.10–3.12 |
| <img src="docs/assets/logos/tafsut.png" width="20" alt=""> | [Tafsut](https://github.com/Tafsut-FM/tafsut) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/nx-ai.png" width="20" alt=""> | [TiRex](https://arxiv.org/abs/2505.23719) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11+ |
| <img src="docs/assets/logos/nixtla.png" width="20" alt=""> | [TimeGPT](https://arxiv.org/abs/2310.03589) | ✓ | ✓ | ✓ | ✓ | ✓ | Requires `NIXTLA_API_KEY` |
| <img src="docs/assets/logos/google.png" width="20" alt=""> | [TimesFM](https://arxiv.org/abs/2310.10688) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/datadog.png" width="20" alt=""> | [Toto](https://arxiv.org/abs/2505.14766) | ✓ | ✓ | ✓ | ✓ | — | |

See the [Model Hub](https://timecopilot.dev/foundationforecast/model-hub/) for version requirements, default quantiles, and family-specific notebooks.

---

## Demo video

<!-- Replace the placeholder below with an embedded demo once the video is ready. -->
<!-- A 60–90 second screencast of the quick example above is enough to show how easy the API is. -->

<p align="center">
  <a href="docs/assets/demo.mp4">
    <img src="docs/assets/demo-thumbnail.png" alt="FoundationForecast demo — coming soon" width="640">
  </a>
</p>

---

## Installation

```bash
pip install foundationforecast
```

Requires Python 3.10+. Some models have additional version requirements — see the [Model Hub](https://timecopilot.dev/foundationforecast/model-hub/).

Optional plotting support:

```bash
pip install "foundationforecast[plot]"
```

---

## Documentation

Full docs, API reference, and example notebooks:

**[timecopilot.dev/foundationforecast](https://timecopilot.dev/foundationforecast/)**

Build and serve docs locally:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

---

## Relationship to TimeCopilot

| | **FoundationForecast** | **TimeCopilot** |
|---|---|---|
| Foundation models (Chronos, Moirai, …) | ✓ | ✓ |
| Unified forecast / CV / anomaly API | ✓ | ✓ |
| Statistical & ML baselines | — | ✓ |
| LLM agent & natural-language queries | — | ✓ |
| Ensembles & distributed inference | — | ✓ |

Use **FoundationForecast** when you only need foundation models. Use **[TimeCopilot](https://github.com/TimeCopilot/timecopilot)** for the full forecasting agent.

---

## Development

```bash
uv sync --group dev --group docs
pre-commit install --install-hooks
uv run pytest
```

---

## License

MIT
