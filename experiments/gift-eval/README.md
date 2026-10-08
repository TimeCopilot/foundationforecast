# FoundationForecast GIFT-Eval Benchmark

End-to-end GIFT-Eval benchmark for [FoundationForecast](https://github.com/AzulGarza/foundationforecast) model wrappers. Evaluation uses [`timecopilot-gift-eval`](https://github.com/TimeCopilot/timecopilot-gift-eval); replication checks compare outputs to official Hugging Face reference CSVs.

## Layout

```
src/
├── eval/       run_gift_eval(), model registry, job config
├── verify/     HF reference loading and replication checks
└── runners/    CLI and Modal entrypoints
tests/          pytest replication checks (imports src.verify)
configs/        models.yaml (full matrix), ci_subset.yaml (CI), replication/ (notebook-aligned)
```

## Setup

```bash
cd experiments/gift-eval
uv sync
```

Installs the in-repo editable `foundationforecast` package from the monorepo root (`../..`), not PyPI — so local runs and CI always use the current wrapper code. After changing the library (for example model weight caching), refresh the lock metadata with `uv lock` in this directory so `uv sync --frozen` in CI matches the editable source version.

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

[`configs/ci_subset.yaml`](configs/ci_subset.yaml) defines **13 jobs**: Chronos on
`m4_weekly/short` and `m4_hourly/short`, plus one representative `model_key` each for
TimesFM 2.5/3.0, TiRex 1.1 and TiRex-2-Zeroshot, Moirai, Toto, FlowState, PatchTST-FM r1,
Granite PatchTST-FM r2, T0, and Tafsut (all on `m4_weekly/short`).
Each job runs on Modal GPU and is **HF-verified** in pytest (metrics must match the
official GIFT-Eval reference CSV).

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
make run-ci
make verify-ci   # sync S3 + pytest
```

## Replication run (notebook-aligned, one model per family)

[`configs/replication/`](configs/replication/) holds **16 families** with params traced to
[official gift-eval notebooks](https://github.com/SalesforceAIResearch/gift-eval/tree/main/notebooks).
Results go to **`s3://foundationforecast-gift-eval/results/replication/<run_id>/`**
(not `results/` or `results/ci/`).

Modal uploads **`foundationforecast/`** (editable install from the monorepo root) plus
`experiments/gift-eval` (`src/`, `configs/`).

1. Pick a run id, e.g. `2026-04-05-nb-v1`.
2. **Pilot** (16 jobs: `m4_weekly` / `short`):

```bash
make run-replication-pilot RUN_ID=2026-04-05-nb-v1
make verify-replication-pilot RUN_ID=2026-04-05-nb-v1
# long full grid in background: DETACHED=1 make run-replication-full RUN_ID=...
```

3. After pilot verify passes, **full grid** (16 × 97 jobs):

Chronos-2 and Chronos Bolt use **foundationforecast** with notebook-aligned inference
(`max_length: null` where the notebook does not cap context; no cross-learning /
`predict_batches_jointly`).

```bash
make run-replication-full RUN_ID=2026-04-05-nb-v1
make verify-replication-full RUN_ID=2026-04-05-nb-v1
```

**Pilot** uses the same **per-job** MASE/CRPS check as CI (`atol=0.01`, `rtol=2.5%`).

**Full grid** uses a **leaderboard aggregate** check per model:
geomean(`MASE/Seasonal_Naive`) and geomean(`CRPS/Seasonal_Naive`) on the same
dataset set as the HF submission, compared with the same default tolerances.
Individual job drift is allowed if the aggregate still matches (see
`src/verify/replication_aggregate.py`). CI pytest (`tests/test_replication.py`)
still enforces strict per-job replication on the CI subset only.

### Replication results (run `2026-10-06-nb-v2`)

FoundationForecast reproduces the **leaderboard aggregates** of every official
submission it wraps: across 15 models with a published reference, the aggregate
MASE and CRPS land within **±1.8%** of the official values (tolerance is 2.5%).
Per-config results are noisier (σ is the standard deviation, over the 97 dataset
configurations, of the per-config difference `ours − official`), but the noise
averages out in the geometric mean the leaderboard ranks on. The whole grid
(16 models × 97 configs) took **33.7 GPU-hours ≈ $37** on a single A10G.

MASE and CRPS are the leaderboard aggregates: geometric mean over the 97 configs of
the metric normalized by Seasonal Naive (lower is better). *Official* is recomputed
from each model's submitted CSV in the GIFT-Eval Hugging Face space; *ours* from this
run. Δ is the relative difference of the aggregates. σ is the standard deviation
across the 97 configs of the per-config difference `ours − official`, computed on the
same Seasonal-Naive-normalized scale as the aggregates. Compute is wall time on an
A10G (Modal) at $1.10/GPU-h. Chronos-2 small has no official submission yet.

| Model | Org. | MASE official | MASE ours | Δ (%) | σ | CRPS official | CRPS ours | Δ (%) | σ | GPU-h | USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TimesFM 3 | Google | 0.667 | 0.675 | +1.2 | 0.021 | 0.456 | 0.462 | +1.3 | 0.015 | 6.73 | 7.41 |
| PatchTST-FM r2 | IBM | 0.685 | 0.684 | -0.1 | 0.028 | 0.467 | 0.467 | +0.0 | 0.030 | 3.31 | 3.64 |
| T0 beta | TFC | 0.687 | 0.689 | +0.3 | 0.010 | 0.474 | 0.475 | +0.4 | 0.014 | 1.41 | 1.55 |
| Tafsut base | Huawei | 0.693 | 0.694 | +0.2 | 0.003 | 0.481 | 0.482 | +0.2 | 0.006 | 1.42 | 1.56 |
| Chronos-2 | Amazon | 0.698 | 0.704 | +0.9 | 0.023 | 0.485 | 0.483 | -0.6 | 0.046 | 1.62 | 1.78 |
| TiRex 2 | NX-AI | 0.697 | 0.704 | +1.0 | 0.015 | 0.478 | 0.485 | +1.6 | 0.025 | 1.29 | 1.42 |
| Toto 2 313M | Datadog | 0.703 | 0.705 | +0.3 | 0.021 | 0.481 | 0.485 | +0.7 | 0.019 | 1.68 | 1.84 |
| PatchTST-FM r1 | IBM | 0.717 | 0.716 | -0.2 | 0.006 | 0.488 | 0.487 | -0.1 | 0.007 | 1.25 | 1.37 |
| TiRex 1.1 | NX-AI | 0.716 | 0.724 | +1.1 | 0.040 | 0.488 | 0.494 | +1.1 | 0.043 | 1.57 | 1.73 |
| Chronos-2 small | Amazon | — | 0.724 | — | — | — | 0.496 | — | — | 0.95 | 1.04 |
| T0 alpha | TFC | 0.724 | 0.729 | +0.7 | 0.018 | 0.494 | 0.495 | +0.3 | 0.017 | 0.94 | 1.03 |
| Moirai 2 small | Salesforce | 0.728 | 0.736 | +1.1 | 0.056 | 0.516 | 0.521 | +0.8 | 0.038 | 2.65 | 2.92 |
| Toto 2 4M | Datadog | 0.757 | 0.761 | +0.6 | 0.017 | 0.524 | 0.531 | +1.3 | 0.028 | 0.55 | 0.61 |
| Chronos Bolt base | Amazon | 0.808 | 0.813 | +0.7 | 0.022 | 0.574 | 0.566 | -1.4 | 0.032 | 4.66 | 5.13 |
| Chronos Bolt small | Amazon | 0.822 | 0.829 | +0.8 | 0.023 | 0.577 | 0.566 | -1.8 | 0.032 | 1.69 | 1.86 |
| Moirai 1.1 large | Salesforce | 0.875 | 0.886 | +1.2 | 0.163 | 0.599 | 0.602 | +0.5 | 0.150 | 1.98 | 2.18 |
| **Total** (16 models) | | | | | | | | | | **33.7** | **37.07** |

<details>
<summary>LaTeX source (needs <code>booktabs</code>)</summary>

```latex
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l rrrr rrrr rr}
\toprule
 & & \multicolumn{4}{c}{MASE$^\dagger$} & \multicolumn{4}{c}{CRPS$^\dagger$} & \multicolumn{2}{c}{Compute} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-12}
Model & Org. & Official & Ours & $\Delta$ (\%) & $\sigma$ & Official & Ours & $\Delta$ (\%) & $\sigma$ & GPU-h & USD \\
\midrule
TimesFM 3 & Google & 0.667 & 0.675 & +1.2 & 0.021 & 0.456 & 0.462 & +1.3 & 0.015 & 6.73 & 7.41 \\
PatchTST-FM r2 & IBM & 0.685 & 0.684 & -0.1 & 0.028 & 0.467 & 0.467 & +0.0 & 0.030 & 3.31 & 3.64 \\
T0 beta & TFC & 0.687 & 0.689 & +0.3 & 0.010 & 0.474 & 0.475 & +0.4 & 0.014 & 1.41 & 1.55 \\
Tafsut base & Huawei & 0.693 & 0.694 & +0.2 & 0.003 & 0.481 & 0.482 & +0.2 & 0.006 & 1.42 & 1.56 \\
Chronos-2 & Amazon & 0.698 & 0.704 & +0.9 & 0.023 & 0.485 & 0.483 & -0.6 & 0.046 & 1.62 & 1.78 \\
TiRex 2 & NX-AI & 0.697 & 0.704 & +1.0 & 0.015 & 0.478 & 0.485 & +1.6 & 0.025 & 1.29 & 1.42 \\
Toto 2 313M & Datadog & 0.703 & 0.705 & +0.3 & 0.021 & 0.481 & 0.485 & +0.7 & 0.019 & 1.68 & 1.84 \\
PatchTST-FM r1 & IBM & 0.717 & 0.716 & -0.2 & 0.006 & 0.488 & 0.487 & -0.1 & 0.007 & 1.25 & 1.37 \\
TiRex 1.1 & NX-AI & 0.716 & 0.724 & +1.1 & 0.040 & 0.488 & 0.494 & +1.1 & 0.043 & 1.57 & 1.73 \\
Chronos-2 small & Amazon & -- & 0.724 & -- & -- & -- & 0.496 & -- & -- & 0.95 & 1.04 \\
T0 alpha & TFC & 0.724 & 0.729 & +0.7 & 0.018 & 0.494 & 0.495 & +0.3 & 0.017 & 0.94 & 1.03 \\
Moirai 2 small & Salesforce & 0.728 & 0.736 & +1.1 & 0.056 & 0.516 & 0.521 & +0.8 & 0.038 & 2.65 & 2.92 \\
Toto 2 4M & Datadog & 0.757 & 0.761 & +0.6 & 0.017 & 0.524 & 0.531 & +1.3 & 0.028 & 0.55 & 0.61 \\
Chronos Bolt base & Amazon & 0.808 & 0.813 & +0.7 & 0.022 & 0.574 & 0.566 & -1.4 & 0.032 & 4.66 & 5.13 \\
Chronos Bolt small & Amazon & 0.822 & 0.829 & +0.8 & 0.023 & 0.577 & 0.566 & -1.8 & 0.032 & 1.69 & 1.86 \\
Moirai 1.1 large & Salesforce & 0.875 & 0.886 & +1.2 & 0.163 & 0.599 & 0.602 & +0.5 & 0.150 & 1.98 & 2.18 \\
\midrule
\multicolumn{10}{l}{Total (16 models, 97 dataset configs each)} & 33.7 & 37.07 \\
\bottomrule
\end{tabular}
\caption{Replication of the GIFT-Eval leaderboard with FoundationForecast (run \texttt{2026-10-06-nb-v2}). $^\dagger$Leaderboard aggregates: geometric mean over the 97 dataset configurations of the metric normalized by Seasonal Naive (lower is better). \emph{Official} is recomputed from the model's submitted CSV in the GIFT-Eval Hugging Face space; \emph{Ours} from our run; $\Delta$ is their relative difference. $\sigma$ is the standard deviation across the 97 configurations of the per-configuration difference $(\text{ours}-\text{official})$ on the same Seasonal-Naive-normalized scale. Compute is wall time on a single NVIDIA A10G (Modal) at \$1.10/GPU-h. Chronos-2 small has no official submission yet.}
\label{tab:gift-eval-replication}
\end{table}
```

</details>

Local single job with replication registry:

```bash
uv run python -m src.runners.run_model --replication \
  --model-key google--timesfm-3.0-pytorch \
  --dataset-name m4_weekly --term short \
  --output-root ./results/replication/local-test
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
metrics), with default tolerances `atol=0.01`, `rtol=0.025`. Other columns in
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
