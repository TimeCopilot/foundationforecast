# FoundationForecast GIFT-Eval Benchmark

End-to-end GIFT-Eval benchmark for [FoundationForecast](https://github.com/AzulGarza/foundationforecast) model wrappers. Evaluation uses [`timecopilot-gift-eval`](https://github.com/TimeCopilot/timecopilot-gift-eval); replication checks compare outputs to official Hugging Face reference CSVs.

## Layout

```
src/
├── eval/       run_gift_eval(), model registry, job config
├── verify/     HF reference loading and replication checks
└── runners/    CLI and Modal entrypoints
tests/          pytest replication checks (imports src.verify)
configs/        models.yaml (full matrix) and ci_subset.yaml (CI)
```

## Setup

```bash
cd experiments/gift-eval
uv sync
```

Installs the in-repo editable `foundationforecast` package from the monorepo root (`../..`), not PyPI — so local runs and CI always use the current wrapper code.

Requires Python 3.11+.

## Dataset

```bash
make download-gift-eval-data
# optional: make upload-data-to-s3
```

## Run a single job locally

```bash
uv run python -m src.runners.run_model \
  --model-key amazon--chronos-bolt-small \
  --dataset-name m4_weekly \
  --term short \
  --storage-path ./data/gift-eval \
  --output-root ./results
```

## CI subset

[`configs/ci_subset.yaml`](configs/ci_subset.yaml) defines **11 jobs**: one representative
`model_key` per FoundationForecast wrapper class (Chronos, TimesFM, TiRex, Moirai, Toto,
FlowState, PatchTST-FM, T0, Sundial, TabPFN), all on `m4_weekly/short`, plus Chronos on
`m4_hourly/short` for a second dataset. Each job runs on Modal GPU and is **HF-verified**
in pytest (metrics must match the official GIFT-Eval reference CSV).

### Local GPU

```bash
uv run python -m src.runners.run_ci --local --verify \
  --storage-path ./data/gift-eval \
  --output-root ./results/ci
```

### Modal (CI / GitHub Actions)

Always re-runs and overwrites results (no skip-if-exists). Full grid skips jobs that
already have outputs.

```bash
uv run modal run -m src.runners.run_modal::run_ci
make sync-ci-results   # download results for local verify / pytest
uv run pytest tests/test_replication.py -n 0 -x
```

## Full benchmark grid (Modal)

One GPU job per `(model_key, dataset, term)`:

```bash
uv run modal run -m src.runners.run_modal::main
```

## Verify against HF references

Compare local/S3 results to official GIFT-Eval CSVs. Uses consolidated
`results/{model_key}/all_results.csv` if present, otherwise aggregates
per-job CSVs under `results/{model_key}/`.

Strict replication asserts **MASE** and **CRPS** only (the GIFT-Eval ranking
metrics), with default tolerances `atol=0.01`, `rtol=0.02`. Other columns in
`all_results.csv` are still written but not compared.

Every verify run also writes a replication analysis table (CSV) with:

| Column | Description |
|--------|-------------|
| `dataset` | GIFT-Eval dataset config (e.g. `m4_weekly/W/short`) |
| `model` | Model alias in results CSV |
| `model_key` | Experiment registry key |
| `time_seconds` | Eval wall time (from per-job `timing.json`) |
| `mase` | Our `eval_metrics/MASE[0.5]` |
| `crps` | Our `eval_metrics/mean_weighted_sum_quantile_loss` |
| `reported_gift_eval_mase` | Official HF reference MASE |
| `reported_gift_eval_crps` | Official HF reference CRPS |
| `mase_diff` | `mase - reported_gift_eval_mase` |
| `crps_diff` | `crps - reported_gift_eval_crps` |

```bash
# CI subset (per-job layout under results/ci/)
uv run python -m src.runners.run_verify --ci

# One model
uv run python -m src.runners.run_verify --model-key amazon--chronos-bolt-small

# All models with a reference_slug in configs/models.yaml
make sync-results   # or: aws s3 sync s3://foundationforecast-gift-eval/results ./results
uv run python -m src.runners.run_verify --all

# Table only (no strict assert) — good for exploratory analysis
uv run python -m src.runners.run_verify --all --verify-only \
  --table-output ./results/replication_table.csv
make replication-table

# Require every HF dataset to be present (not just compare overlap)
uv run python -m src.runners.run_verify --all --require-complete
```

Or in one step:

```bash
make verify-all
```

**Note:** `time_seconds` is recorded when a job runs via `run_gift_eval` (writes
`timing.json` next to each `all_results.csv`). To backfill timing for jobs that
ran before timing was added:

```bash
# Full grid: rerun only jobs with results but no timing.json on S3
uv run modal run -m src.runners.run_modal::run_missing_timing

# Full grid: force rerun everything (also refreshes metrics)
uv run modal run -m src.runners.run_modal::main --force

# CI subset locally
uv run python -m src.runners.run_ci --local --missing-timing-only

# CI on Modal always reruns with force=True (timing included every CI run)
uv run modal run -m src.runners.run_modal::run_ci
```

Then sync and rebuild the table:

```bash
make sync-results
make replication-table
```

## Consolidate S3 results

```bash
uv run python -m src.runners.download_results --model-key amazon--chronos-bolt-small
```

## Infrastructure

- **S3 bucket:** `foundationforecast-gift-eval`
- **Modal secrets:**
  - `aws-secret` — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  - `hf-secret` — `HF_TOKEN` (required for gated models like `t0-alpha`; create with
    `modal secret create hf-secret HF_TOKEN=hf_...`)
- **Modal tokens:** `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
- **Hugging Face:** accept model licenses on the Hub, then set `HF_TOKEN` in `hf-secret`

## Adding a model

1. Add an entry to `configs/models.yaml` with slugified `repo_id` as `model_key`
   (`org--model`), `class`, `kwargs.repo_id`, and `reference_slug`.
2. Set `kwargs.repo_id` from the official GIFT-Eval
   `results/{reference_slug}/config.json` → `model_link` (many models use
   gifteval-specific HF repos, not the default public checkpoint).
3. Set `reference_slug` to the official GIFT-Eval folder name; `alias` defaults from
   that in `build_model()` — set `kwargs.alias` explicitly when the CSV `model` column
   differs from the folder slug (e.g. `chronos_base` → `Chronos_base`).
4. Set `reference_slug: null` if no public reference exists (verify skips that model).
