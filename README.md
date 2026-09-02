*One API for the time series foundation era · Forecast · Cross-validation · Anomaly detection*

---

## One API for the time series foundation era

Forecasting and time series have entered their foundation era. But, just like in the LLM space, different models bring different inductive biases and perform differently across domains, datasets, and even horizons.

The world already uses different LLMs for different use cases. We're seeing the same thing in forecasting: there is no single model that dominates everywhere. Results change with the data distribution and forecasting horizon, as we've seen in [Impermanent](https://arxiv.org/abs/2603.08707) and other benchmarks such as [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [FEV](https://arxiv.org/abs/2509.26468).

But every lab also ships its own API, dependencies, data conventions, and learning curve. That fragmentation makes foundation models hard to compare fairly, and even harder to use together in production.

**FoundationForecast** removes that friction: one `FoundationForecast` class, one data format, and the same methods: `forecast`, `cross_validation`, and `detect_anomalies`, across the leading time series foundation models. ✨

Developed with 💙 by the [TimeCopilot](https://timecopilot.dev/) team.

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

Your DataFrame needs three columns: `unique_id`, `ds`, and `y`. Dates in `ds` are parsed automatically; no need to pass `parse_dates`.

---

## Supported models

Every model supports **forecast**, **cross-validation**, and **anomaly detection** through the same API. **Intervals** means prediction intervals via `level` or quantile forecasts. **Finetuning** marks models that can adapt to your data at inference time. **License** is the [weight/checkpoint license](https://huggingface.co/models) on the default Hugging Face repo (or provider terms for hosted APIs). See the note below for production use.

Pass any Hugging Face `repo_id` (or local checkpoint path) supported by the underlying model class.


|     | Model                                                       | Forecast | CV  | Anomalies | Intervals | Finetuning | License                |
| --- | ----------------------------------------------------------- | -------- | --- | --------- | --------- | ---------- | ---------------------- |
|     | [Chronos](https://arxiv.org/abs/2403.07815)                 | ✓        | ✓   | ✓         | ✓         | ✓          | Apache-2.0             |
|     | [FlowState](https://arxiv.org/abs/2508.05287)               | ✓        | ✓   | ✓         | ✓         |            | Apache-2.0             |
|     | [Moirai](https://arxiv.org/abs/2402.02592)                  | ✓        | ✓   | ✓         | ✓         |            | CC-BY-NC-4.0           |
|     | [PatchTST-FM](https://arxiv.org/abs/2602.06909)             | ✓        | ✓   | ✓         | ✓         |            | CC-BY-NC-SA-4.0        |
|     | [Sundial](https://arxiv.org/abs/2502.00816)                 | ✓        | ✓   | ✓         | ✓         |            | Apache-2.0             |
|     | [T0](https://huggingface.co/theforecastingcompany/t0-alpha) | ✓        | ✓   | ✓         | ✓         |            | Apache-2.0†            |
|     | [TabPFN](https://arxiv.org/abs/2501.02945)                  | ✓        | ✓   | ✓         | ✓         |            | TabPFN NC‡             |
|     | [Tafsut](https://github.com/Tafsut-FM/tafsut)               | ✓        | ✓   | ✓         | ✓         |            | MIT                    |
|     | [TiRex](https://arxiv.org/abs/2505.23719)                   | ✓        | ✓   | ✓         | ✓         |            | Community / Apache-2.0 |
|     | [TimeGPT](https://arxiv.org/abs/2310.03589)                 | ✓        | ✓   | ✓         | ✓         | ✓          | Nixtla API§            |
|     | [TimesFM](https://arxiv.org/abs/2310.10688)                 | ✓        | ✓   | ✓         | ✓         |            | Apache-2.0             |
|     | [Toto](https://arxiv.org/abs/2505.14766)                    | ✓        | ✓   | ✓         | ✓         |            | Apache-2.0             |


Licenses verified against Hugging Face model cards (March 2026). Check the model card for your `repo_id` when in doubt.

**What this means for production**

- **Apache-2.0**, **MIT**: generally fine for commercial production (retain notices; T0† is also gated on Hugging Face: accept terms and set `HF_TOKEN`).
- **CC-BY-NC-4.0** (Moirai), **CC-BY-NC-SA-4.0** (PatchTST-FM): **non-commercial** only; not for revenue-generating production without a separate agreement from the rights holder.
- **TabPFN NC**‡: TabPFN-2.6+ weights are non-commercial; production requires a [Prior Labs commercial license or API](https://docs.priorlabs.ai/models). First use also requires accepting terms at [ux.priorlabs.ai](https://ux.priorlabs.ai) (`TABPFN_TOKEN`).
- **Community / Apache-2.0** (TiRex): TiRex 1.0 uses the [NXAI Community License](https://huggingface.co/NX-AI/TiRex/blob/main/LICENSE) (commercial limits for large enterprises); TiRex 2.0 is Apache-2.0.
- **Nixtla API**§: hosted service via `NIXTLA_API_KEY`; production under [Nixtla terms/pricing](https://www.nixtla.io/docs), not open weights.

**FoundationForecast** itself is [Apache-2.0](LICENSE) regardless of which model you plug in.



Some models require specific Python versions (e.g. FlowState 3.11-3.13, TabPFN < 3.13). See the [Model Hub](https://timecopilot.dev/foundationforecast/model-hub/) for details and default checkpoints.

**Example checkpoints & API model IDs**

- **Chronos:** `amazon/chronos-t5-{tiny,mini,small,base,large}`, `amazon/chronos-bolt-{tiny,mini,small,base}`, `amazon/chronos-2`
- **FlowState:** `ibm-research/flowstate`, `ibm-granite/granite-timeseries-flowstate-r1`
- **Moirai:** `Salesforce/moirai-{1.0,1.1,2.0}-R-{small,base,large}`, `Salesforce/moirai-moe-1.0-R-`*
- **PatchTST-FM:** `ibm-research/patchtst-fm-r1`
- **Sundial:** `thuml/sundial-base-128m`
- **T0:** `theforecastingcompany/t0-alpha`
- **TabPFN:** `tabpfn-local`, `tabpfn-client`
- **Tafsut:** `Tafsut-FM/tafsut-univariate-base`
- **TiRex:** `NX-AI/TiRex`, `NX-AI/TiRex-2`
- **TimeGPT:** pass `model=` to `TimeGPT()`, e.g. `timegpt-1`, `[timegpt-1-long-horizon](https://www.nixtla.io/docs/forecasting/model-version/longhorizon_model)`, `timegpt-2-mini`, `timegpt-2`, `timegpt-2-pro`
- **TimesFM:** `google/timesfm-{1.0-200m,2.0-500m,2.5-200m}-pytorch`
- **Toto:** `Datadog/Toto-Open-Base-1.0`, `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}`



---

## Installation

**Recommended:** [uv](https://docs.astral.sh/uv/) installs fast, locks dependencies reproducibly, and matches how this repo is developed and tested (especially useful with heavy ML stacks like torch and transformers).

```bash
uv add foundationforecast
```

Or with pip:

```bash
pip install foundationforecast
```

Requires Python 3.10+. Some models have additional version requirements; see the [Model Hub](https://timecopilot.dev/foundationforecast/model-hub/).

Optional plotting support:

```bash
uv add "foundationforecast[plot]"
# or: pip install "foundationforecast[plot]"
```

---

## Relationship to TimeCopilot


|                                        | **FoundationForecast** | **TimeCopilot** |
| -------------------------------------- | ---------------------- | --------------- |
| Foundation models (Chronos, Moirai, …) | ✓                      | ✓               |
| Unified forecast / CV / anomaly API    | ✓                      | ✓               |
| Statistical & ML baselines             |                        | ✓               |
| LLM agent & natural-language queries   |                        | ✓               |
| Ensembles & distributed inference      |                        | ✓               |


Use **FoundationForecast** when you only need foundation models. Use **[TimeCopilot](https://github.com/TimeCopilot/timecopilot)** for the full forecasting agent.

---

## License

**FoundationForecast** is licensed under [Apache License 2.0](LICENSE). You may use, modify, and deploy the library in production, including commercial applications.

Model weights and hosted APIs have separate licenses. See the **License** column and production note in [Supported models](#supported-models) above.