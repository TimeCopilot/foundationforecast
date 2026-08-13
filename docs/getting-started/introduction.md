# Introduction

**foundationforecast** is a focused library for running state-of-the-art **foundation time series models** through a unified Python API. It extracts the foundation-model layer from [TimeCopilot](https://github.com/TimeCopilot/timecopilot) so you can forecast, cross-validate, and detect anomalies without statistical, ML, or agent dependencies.

## What you can do

- Run multiple pretrained foundation models with one interface via [`FoundationForecast`](../api/forecaster.md).
- Compare Chronos, Moirai, TimesFM, Toto, TiRex, TimeGPT, FlowState, and more side by side.
- Finetune supported models (Chronos 2, TimeGPT) on your own data.
- Detect anomalies using cross-validated z-score tests on any forecaster.

## Supported models

See the [Model Hub](../model-hub.md) for the full list and Python version requirements. Some models require Python 3.11+ (TiRex) or have upper version bounds (TabPFN, T0, FlowState).

## Relationship to TimeCopilot

TimeCopilot adds LLM agents, classical stats/ML models, ensembles, and distributed scaling on top of this foundation layer. Use **foundationforecast** when you only need foundation models; use **TimeCopilot** for the full forecasting agent and broader model hub.
