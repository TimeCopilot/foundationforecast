<div align="center">
  <img src="docs/assets/logo-dark.svg#gh-dark-mode-only" alt="FoundationForecast" width="900">
  <img src="docs/assets/logo-light.svg#gh-light-mode-only" alt="FoundationForecast" width="900">
</div>
<div align="center">
  <em>One API for every time series foundation model · Forecast · Cross-validate · Detect anomalies</em>
</div>
<div align="center">
  <a href="https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml"><img src="https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/foundationforecast"><img src="https://img.shields.io/pypi/v/foundationforecast.svg" alt="PyPI"></a>
  <a href="https://github.com/TimeCopilot/foundationforecast"><img src="https://img.shields.io/pypi/pyversions/foundationforecast.svg" alt="versions"></a>
  <a href="https://github.com/TimeCopilot/foundationforecast/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TimeCopilot/foundationforecast" alt="license"></a>
  <a href="https://discord.gg/7GEdHR6Pfg"><img src="https://img.shields.io/discord/1387291858513821776?label=discord" alt="Join Discord"></a>
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

Pass any Hugging Face `repo_id` (or local checkpoint path) supported by the underlying model class.

| | Model | Forecast | CV | Anomalies | Intervals | Finetune | Notes |
|:-:|---|:-:|:-:|:-:|:-:|:-:|---|
| <img src="docs/assets/logos/amazon.png" width="30" alt=""> | [Chronos](https://arxiv.org/abs/2403.07815) | ✓ | ✓ | ✓ | ✓ | ✓ | Finetune on Chronos 2 |
| <img src="docs/assets/logos/ibm.png" width="30" alt=""> | [FlowState](https://arxiv.org/abs/2508.05287) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/salesforce.png" width="30" alt=""> | [Moirai](https://arxiv.org/abs/2402.02592) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/ibm.png" width="30" alt=""> | [PatchTST-FM](https://arxiv.org/abs/2602.06909) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/thuml.png" width="30" alt=""> | [Sundial](https://arxiv.org/abs/2502.00816) | ✓ | ✓ | ✓ | ✓ | — | Python 3.10–3.12 |
| <img src="docs/assets/logos/tfc.png" width="30" alt=""> | [T0](https://huggingface.co/theforecastingcompany/t0-alpha) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11–3.13 |
| <img src="docs/assets/logos/priorlabs.png" width="30" alt=""> | [TabPFN](https://arxiv.org/abs/2501.02945) | ✓ | ✓ | ✓ | ✓ | — | Python 3.10–3.12 |
| <img src="docs/assets/logos/tafsut.png" width="30" alt=""> | [Tafsut](https://github.com/Tafsut-FM/tafsut) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/nx-ai.png" width="30" alt=""> | [TiRex](https://arxiv.org/abs/2505.23719) | ✓ | ✓ | ✓ | ✓ | — | Python 3.11+ |
| <img src="docs/assets/logos/nixtla.png" width="30" alt=""> | [TimeGPT](https://arxiv.org/abs/2310.03589) | ✓ | ✓ | ✓ | ✓ | ✓ | Requires `NIXTLA_API_KEY` |
| <img src="docs/assets/logos/google.png" width="30" alt=""> | [TimesFM](https://arxiv.org/abs/2310.10688) | ✓ | ✓ | ✓ | ✓ | — | |
| <img src="docs/assets/logos/datadog.png" width="30" alt=""> | [Toto](https://arxiv.org/abs/2505.14766) | ✓ | ✓ | ✓ | ✓ | — | |

### Checkpoints & weights

<details>
<summary><strong>Chronos</strong> — default <code>amazon/chronos-t5-large</code> (710M)</summary>

- **Chronos 2:** `amazon/chronos-2`, `autogluon/chronos-2-synth`, `autogluon/chronos-2-small`
- **Chronos-Bolt:** `amazon/chronos-bolt-tiny` … `amazon/chronos-bolt-base` (9M–205M)
- **Chronos-T5:** `amazon/chronos-t5-tiny` … `amazon/chronos-t5-large` (8M–710M)

</details>

<details>
<summary><strong>FlowState</strong> — default <code>ibm-research/flowstate</code></summary>

- `ibm-research/flowstate`
- `ibm-granite/granite-timeseries-flowstate-r1`

</details>

<details>
<summary><strong>Moirai</strong> — default <code>Salesforce/moirai-1.0-R-large</code></summary>

- **Moirai 1.0 / 1.1:** `Salesforce/moirai-1.0-R-{small,base,large}`, `Salesforce/moirai-1.1-R-{small,base,large}`
- **Moirai 2.0:** `Salesforce/moirai-2.0-R-{small,base,large}`
- **Moirai MoE:** `Salesforce/moirai-moe-1.0-R-{small,base,large}`

[Full collection →](https://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742)

</details>

<details>
<summary><strong>PatchTST-FM</strong> — default <code>ibm-research/patchtst-fm-r1</code></summary>

- `ibm-research/patchtst-fm-r1`

</details>

<details>
<summary><strong>Sundial</strong> — default <code>thuml/sundial-base-128m</code></summary>

- `thuml/sundial-base-128m`

</details>

<details>
<summary><strong>T0</strong> — default <code>theforecastingcompany/t0-alpha</code> (~102M)</summary>

- `theforecastingcompany/t0-alpha`

</details>

<details>
<summary><strong>TabPFN</strong> — TabPFN-v2 via tabpfn-time-series</summary>

- Local (`tabpfn-local`) or cloud (`tabpfn-client`) inference modes

</details>

<details>
<summary><strong>Tafsut</strong> — default <code>Tafsut-FM/tafsut-univariate-base</code></summary>

- `Tafsut-FM/tafsut-univariate-base`

</details>

<details>
<summary><strong>TiRex</strong> — default <code>NX-AI/TiRex</code></summary>

- **TiRex 1.0:** `NX-AI/TiRex`
- **TiRex 2.0:** `NX-AI/TiRex-2`

</details>

<details>
<summary><strong>TimeGPT</strong> — default <code>timegpt-1</code> (API)</summary>

- `timegpt-1` and other versions via the [Nixtla API](https://www.nixtla.io/docs)

</details>

<details>
<summary><strong>TimesFM</strong> — default <code>google/timesfm-2.0-500m-pytorch</code></summary>

- `google/timesfm-1.0-200m-pytorch`
- `google/timesfm-2.0-500m-pytorch`
- `google/timesfm-2.5-200m-pytorch`

</details>

<details>
<summary><strong>Toto</strong> — default <code>Datadog/Toto-Open-Base-1.0</code> (151M)</summary>

- **Toto 1.0:** `Datadog/Toto-Open-Base-1.0`
- **Toto 2.0:** `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}`

</details>

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

Apache License 2.0 — see [LICENSE](LICENSE).
