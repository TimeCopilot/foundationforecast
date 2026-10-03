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

`configs/models.yaml` lists every **native futr exog** wrapper that has a fev-bench results CSV: **Chronos-2**, **TimesFM-3**, **T0-beta**, **TiRex-2** (not TimeGPT — no official fev-bench row file).

Modal CI runs **each** of those models on two known-only tasks:

- **`epf_pjm`** — 1 series, h=24  
- **`entsoe_1H`** — 6 series, h=168  

Replication compares `test_error` to the matching row in each model’s leaderboard CSV.

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
