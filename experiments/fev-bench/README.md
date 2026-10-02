# fev-bench experiment

Run [fev-bench](https://arxiv.org/abs/2509.26468) tasks through **foundationforecast** with **known-future dynamic covariates only**.

## Scope: known-dynamic-only (n=13)

fev-bench defines `known_dynamic_columns`, `past_dynamic_columns`, and `static_columns`. FoundationForecast iteration 1 supports future-known dynamic exog (`X_df` / CV columns) only. This experiment runs the **13 tasks** where covariates are exclusively known-dynamic (no past-dynamic or static columns).

Do **not** compare aggregate results to the paper’s “30 known dynamic” or “42 dynamic covariate” figures without noting that those mixes include tasks where leaderboard models used past/static covariates we do not feed.

Task list: [configs/known_only_tasks.yaml](configs/known_only_tasks.yaml).

## Setup

```bash
cd experiments/fev-bench
uv sync
```

Requires Hugging Face access to `autogluon/fev_datasets` (`HF_TOKEN` recommended).

## Run a single job locally

```bash
uv run python -m src.runners.run_model \
  --model-key amazon--chronos-2 \
  --task-name epf_de
```

## CI (Modal)

```bash
uv run modal run -m src.runners.run_modal::run_ci
make sync-ci-results
uv run pytest tests/test_replication.py -n 0 -x
```

## Models and leaderboard replication

`configs/models.yaml` includes **Chronos-2** and **TimesFM-3** (`google/timesfm-3.0-pytorch`), matching fev-bench results `chronos-2.csv` and `timesfm-3.csv`. CI runs **Chronos-2 only** on `epf_de` and `entsoe_1H` to limit Modal cost; add TimesFM jobs to `ci_subset.yaml` when you want the same checks for TimesFM-3.

## Full covariate grid (13 tasks × models in `configs/models.yaml`)

```bash
uv run modal run -m src.runners.run_modal::main
```

## Outputs

Per job under `results/{model_key}/{task_name}/`:

- `summary.json` — fev `evaluation_summary` dict
- `summary.csv` — one-row table for verification
- `timing.json` — wall-clock seconds for the forecast loop

S3 bucket: `foundationforecast-fev-bench` (`results/ci/` for CI, `results/` for full grid).
