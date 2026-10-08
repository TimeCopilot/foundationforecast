# FoundationForecast replicates Salesforce's GIFT-Eval for $37

Replication of **Salesforce's full GIFT-Eval** forecasting benchmark **under a single API (FoundationForecast)**. The full replication of **16 models** cost **$37** and took **≈34 hours** of GPU time. The best performance among the implemented models is achieved by **TimesFM 3 (Google) for under $8**.

This directory also [regression-tests](https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml) FoundationForecast on every merge to `main`.

## Why?

Companies and practitioners rely on accurate forecasts to make better decisions. Some of those decisions have to be made **quickly**; as compute capacity grows, that speed requirement is even sharper in nascent, high-frequency settings such as energy markets, cloud capacity, and real-time operations. Other decisions have to be made **under a budget**, so accuracy should be judged together with computational cost.

**Time Series Foundation Models (TSFMs)** have changed how forecasting is done. What is still missing is a clear view of the **accuracy–compute trade-off**, and a **unified way** to evaluate the models and deploy them in production.

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) ([Aksu et al., 2024](https://arxiv.org/abs/2410.10393)) is one of the benchmarks practitioners and researchers use to publish SOTA results, by uploading their own implementations. The process does not, by itself, give a unified path to replicability: each model ships with its own API, inference rules, and design. The leaderboard also reports **accuracy only**. It does not measure computational cost on shared infrastructure.

We built this reproducible experiment to show that a **unified API** makes it practical to test both **accuracy and cost** on large time series data. **FoundationForecast** is that API. The results below are from running it on the full GIFT-Eval grid.

## Results

We replicated **16 foundation models** on Salesforce's full GIFT-Eval: 23 datasets, ~144k time series, 177M data points, evaluated as **326,490** forecasting tasks (one task = one series test window; multivariate series are forecast per variate, giving **371,330** univariate forecasts) grouped into **97** dataset / frequency / horizon configurations. That is **5.9 billion** probabilistic forecast values (forecasts × horizon × 16 models × 9 quantiles).

**MASE** (point) and **CRPS** (probabilistic) follow the leaderboard convention: each configuration is normalized by Seasonal Naive and aggregated with a **geometric mean** across the 97 configurations. Cost is **not** part of the original evaluation; we estimate it from per-job GPU wall time on a single **NVIDIA A10G** at **$1.10/GPU-hour**.

Every FoundationForecast implementation reproduces the model's aggregated MASE and CRPS **within 2%** of the official leaderboard submission (Chronos-2 small has no published reference yet).

### Replication

The table below compares **official** GIFT-Eval aggregates with **FF** (the same aggregate from our FoundationForecast run). Organizations are grouped alphabetically. Gold / silver / bronze mark the **1st / 2nd / 3rd best CRPS** in the Official and FF columns — the ranking is the same on both sides: **TimesFM 3**, **PatchTST-FM r2**, **T0 beta**.

<img src="https://github.com/user-attachments/assets/627c11c8-cfce-4e40-ba5d-3c3470d47c8c" alt="GIFT-Eval replication table: official vs FoundationForecast MASE and CRPS, with gold/silver/bronze on the top-3 CRPS models" width="1100">

### Pareto frontier

Accuracy alone does not decide which model to deploy. The figure below plots **cost vs accuracy** for the same 16 models: cheaper is to the right, better (lower error) is up. The purple polyline is the **Pareto frontier**. TimesFM 3 is the most accurate model we ran, at **$7.41**; Toto 2 4M is the cheapest plotted point, at **$0.61**. The full grid is **$37 / 34 GPU-hours**.

<img src="https://github.com/user-attachments/assets/ed6415a6-f523-4969-85e1-4055d678087f" alt="Cost vs accuracy Pareto frontier on 5.9B probabilistic forecast values for 16 foundation models on GIFT-Eval" width="1100">

## Reproducibility

```bash
cd experiments/gift-eval
uv sync
make download-gift-eval-data

# 16 models × 97 configs
make run-replication-full RUN_ID=2026-10-06-nb-v2
make verify-replication-full RUN_ID=2026-10-06-nb-v2
```

Params follow the [official GIFT-Eval notebooks](https://github.com/SalesforceAIResearch/gift-eval/tree/main/notebooks). Results land in `s3://foundationforecast-gift-eval/results/replication/<run_id>/`. The full grid skips jobs that already have outputs unless `FORCE=1`. Long runs: `DETACHED=1`. Parallel GPUs: `MAX_CONTAINERS=20` (default).

On every merge to `main`, [CI](https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml) re-runs a 13-job subset (`make run-ci` / `make verify-ci`) with **strict per-job** MASE/CRPS against the Hugging Face references.

### Infrastructure

| Need | What |
| --- | --- |
| Python | 3.11+ (`uv sync` from this directory; editable install of repo-root `foundationforecast`) |
| GPU jobs | [Modal](https://modal.com) (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) |
| Object store | S3 bucket `foundationforecast-gift-eval` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| Weights | `HF_TOKEN` (gated models such as `t0-alpha`); accept Hub licenses |
| Optional | `DETACHED=1`, `FORCE=1`, `MAX_CONTAINERS=N` |

Modal secrets: `aws-secret`, `hf-secret` (`modal secret create hf-secret HF_TOKEN=hf_...`).

## Conclusion

GIFT-Eval already tells us **which models are accurate**. Running them through one API, on one GPU SKU, adds the missing axis: **what that accuracy costs**. The replication shows the leaderboard numbers are recoverable (aggregates within 2%). The Pareto plot shows they are not interchangeable: the most accurate model in this run is also one of the more expensive, and several cheaper models sit close enough on CRPS to matter when the budget is the constraint.

If you are choosing a TSFM for production, measure **both**. FoundationForecast is the API we used to make that measurement reproducible.

## Acknowledgements

Thanks to the teams who trained and released the models in this grid: Amazon (Chronos / Chronos-2 / Chronos Bolt), Datadog (Toto 2), Google (TimesFM 3), Huawei (Tafsut), IBM (PatchTST-FM), NX-AI (TiRex), Salesforce (Moirai), and The Forecasting Company (T0). Thanks also to Salesforce AI Research for GIFT-Eval — the datasets, notebooks, and public leaderboard this experiment replicates.

## References

- Aksu, T., Woo, G., Liu, J., Liu, X., Liu, C., Savarese, S., Xiong, C., & Sahoo, D. (2024). *GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation*. [arXiv:2410.10393](https://arxiv.org/abs/2410.10393).
- Ansari, A. F., et al. (2024). *Chronos: Learning the Language of Time Series*. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).
- Das, A., Kong, W., Sen, R., & Zhou, Y. (2023). *A decoder-only foundation model for time-series forecasting* (TimesFM). [arXiv:2310.10688](https://arxiv.org/abs/2310.10688).
- Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D. (2024). *Unified Training of Universal Time Series Forecasting Transformers* (Moirai). [arXiv:2402.02592](https://arxiv.org/abs/2402.02592).
- Cohen, B., et al. (2025). *This Time is Different: An Observability Perspective on Time Series Foundation Models* (Toto). [arXiv:2505.14766](https://arxiv.org/abs/2505.14766).
- Auer, A., et al. (2025). *TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning*. [arXiv:2505.23719](https://arxiv.org/abs/2505.23719).
- Auer, A., et al. (2026). *TiRex-2*. [arXiv:2607.01204](https://arxiv.org/abs/2607.01204).
- IBM. *PatchTST-FM*. [arXiv:2602.06909](https://arxiv.org/abs/2602.06909).
- The Forecasting Company. [T0 alpha](https://huggingface.co/theforecastingcompany/t0-alpha) / [T0 beta](https://huggingface.co/theforecastingcompany/t0-beta).
- Tafsut-FM. [Tafsut univariate base](https://huggingface.co/Tafsut-FM/tafsut-univariate-base).
- Garza, A., & Rosillo, R. (2026). *FoundationForecast: The API for time series foundation models*. <https://github.com/TimeCopilot/foundationforecast>.

## How to cite

```bibtex
@software{foundationforecast-gift-eval,
  title = {FoundationForecast replicates Salesforce's GIFT-Eval for \$37},
  author = {Garza, Azul and Rosillo, Ren{\'e}e},
  year = {2026},
  url = {https://github.com/TimeCopilot/foundationforecast/tree/main/experiments/gift-eval},
  note = {Replication of 16 time series foundation models on GIFT-Eval under a single API},
}
```

```bibtex
@software{foundationforecast,
  title = {FoundationForecast: The API for time series foundation models},
  author = {Garza, Azul and Rosillo, Ren{\'e}e},
  year = {2026},
  url = {https://github.com/TimeCopilot/foundationforecast},
  license = {Apache-2.0},
}
```
