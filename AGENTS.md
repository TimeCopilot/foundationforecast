# Agent guide

FoundationForecast is a Python library: one `FoundationForecast` API over many time-series foundation model wrappers. Human-oriented setup lives in [`docs/contributing.md`](docs/contributing.md).

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): subject`.

- One **type** per commit (`feat`, `fix`, `test`, `docs`, `refactor`, `chore`, …).
- Prefer separate commits by type (implementation, then tests, then docs)—not one mixed commit per change.

## Releases

Releases are **SemVer** tags on `main`. There is no rolling `CHANGELOG.md`; each version gets its own file under [`docs/changelogs/`](docs/changelogs/) (see [`v0.1.11.md`](docs/changelogs/v0.1.11.md) for structure: Features, Fixes, Documentation, Tooling, compare link).

1. Add `docs/changelogs/vMAJOR.MINOR.PATCH.md` and link it at the top of [`docs/changelogs/index.md`](docs/changelogs/index.md).
2. Register the file in [`mkdocs.yml`](mkdocs.yml) under Changelogs (newest first).
3. Bump [`pyproject.toml`](pyproject.toml) `[project].version` (single source of truth; no `__version__` in the package).
4. Run `uv lock`. If dependencies or the editable `foundationforecast` pin changed for benchmarks, refresh locks under [`experiments/gift-eval`](experiments/gift-eval) and/or [`experiments/fev-bench`](experiments/fev-bench) with `uv lock` there too.
5. Update user-visible docs for the release when behavior or the model surface changes: [`README.md`](README.md) (support matrix, highlights timeline), [`docs/model-hub.md`](docs/model-hub.md), API pages under `docs/api/`, and examples/notebooks as needed.
6. Open a PR titled **`chore(release): vX.Y.Z`**, merge to `main`.
7. Publish from `main` (creates the tag and GitHub Release):

   ```bash
   gh release create vX.Y.Z --discussion-category "General" -F docs/changelogs/vX.Y.Z.md
   ```

Pushing the release tag runs:

- [`.github/workflows/release.yaml`](.github/workflows/release.yaml) — ensures the GitHub Release exists, then PyPI publish (`uv build` / `uv publish` with trusted publishing) and post-publish import smoke tests.
- [`.github/workflows/notify-release.yaml`](.github/workflows/notify-release.yaml) — Discord notification with the same changelog body.

## Tests as documentation

Behavior changes need tests that fail without the change. Treat CI layout as the contract for what must stay green.

### Default local run

```bash
uv sync --group dev --group docs
uv run pytest          # excludes `docs` and `benchmark` markers (see pyproject.toml)
uv run pre-commit run --all-files
```

Docs and docstring examples:

```bash
uv run pytest -m docs -o addopts= -n 0
uv run --group docs mkdocs build
```

### What CI enforces ([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml))

| Job | Purpose |
| --- | --- |
| **test** | `uv run pytest` on Python 3.10–3.13; coverage gate (≥80%) on 3.12. Needs `HF_TOKEN` / `TABPFN_TOKEN` for model tests. |
| **test-docs** | Executable markdown in `docs/` (except changelog files), `README.md`, and Python docstrings via `mktestdocs`. |
| **build-docs** | `mkdocs build` |
| **lint** | Full `pre-commit` (ruff, mypy on `foundationforecast/` only—`experiments/` excluded) |
| **test-gift-eval** | Modal GPU run for [`configs/ci_subset.yaml`](experiments/gift-eval/configs/ci_subset.yaml), then `make verify-ci` (S3 sync + strict per-job MASE/CRPS vs Hugging Face references). |
| **test-fev-bench** | Modal CI for known-dynamic tasks, S3 sync, then `tests/test_replication.py` vs fev-bench leaderboard CSVs. |

Performance regressions: [`.github/workflows/codspeed.yml`](.github/workflows/codspeed.yml) (`pytest -m benchmark` under `tests/benchmarks/`).

### Cross-model contract ([`tests/models/test_models.py`](tests/models/test_models.py))

Public behavior on model wrappers is exercised **for every entry in** [`tests/models/conftest.py`](tests/models/conftest.py) `models` via `@pytest.mark.parametrize("model", models)`.

- **New cross-cutting feature** (e.g. known-future exog, `level`, quantiles, date handling): add or extend parametrized tests in `test_models.py` so each model runs the scenario. Use capability checks and `pytest.skip` when a checkpoint legitimately does not support the feature (see `supports_native_futr_exog()` in the exog tests)—do not omit models from the parametrization.
- **New public model**: append representative instances to `conftest.py` `models` so existing `test_models.py` coverage applies automatically; add model-specific tests under `tests/models/test_<name>.py` when needed. Follow [`docs/contributing.md`](docs/contributing.md) for exports, model hub, and API docs.

Also add focused tests where the API is not model-specific: `tests/core/`, `tests/test_foundation_forecast.py`. Use `@pytest.mark.models` when tests download weights (CI caches Hugging Face and TabPFN checkpoints).

**Replication** (leaderboard MASE/CRPS or fev-bench `test_error`): extend [`experiments/gift-eval`](experiments/gift-eval) and/or [`experiments/fev-bench`](experiments/fev-bench) when a wrapper must match an official submission—not a substitute for `test_models.py`.

## Repository layout (short)

- `foundationforecast/` — library (`core/`, `models/`).
- `tests/` — unit, integration, docs, and CodSpeed benchmarks.
- `docs/` — MkDocs site; changelogs in `docs/changelogs/v*.md`.
- `experiments/gift-eval/`, `experiments/fev-bench/` — Modal + S3 replication harnesses (editable install of repo root).
