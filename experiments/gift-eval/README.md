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

Requires Python 3.11+.

## Dataset

```bash
make download-gift-eval-data
# optional: make upload-data-to-s3
```

## Run a single job locally

```bash
uv run python -m src.runners.run_model \
  --model-key chronos-bolt-small \
  --dataset-name m4_weekly \
  --term short \
  --storage-path ./data/gift-eval \
  --output-root ./results
```

## CI subset (local GPU)

```bash
uv run python -m src.runners.run_ci --local --verify \
  --storage-path ./data/gift-eval \
  --output-root ./results/ci
```

## CI subset (Modal)

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

## Consolidate S3 results

```bash
uv run python -m src.runners.download_results --model-key chronos-bolt-small
```

## Infrastructure

- **S3 bucket:** `foundationforecast-gift-eval`
- **Modal secret:** `aws-secret` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- **Modal tokens:** `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
- **Hugging Face:** `HF_TOKEN` (dataset + model weights)

## Adding a model

1. Add an entry to `configs/models.yaml` with `class`, `kwargs`, and `reference_slug`.
2. Match `alias` to the official GIFT-Eval `model` column in the HF results CSV.
3. Set `reference_slug: null` if no public reference exists (verify skips that model).
