Your contributions are highly appreciated!

## Prerequisites

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Install [pre-commit](https://pre-commit.com/#install).

!!! tip "Tip"
    Once `uv` is installed, you can install `pre-commit` with:

    ```
    uv tool install pre-commit
    ```

- Set up the pre-commit hook:

    ```
    pre-commit install --install-hooks
    ```

## Installation and Setup

To run **foundationforecast** in your local environment:

1. Fork and clone the repository:

    ```
    git clone git@github.com:<your-username>/foundationforecast.git
    ```

2. Navigate into the project folder:

    ```
    cd foundationforecast
    ```

3. Install dependencies:

    ```
    uv sync --group dev --group docs
    ```

✅ You're ready to start contributing!

## Python 3.14 and 3.15 (preview)

The library installs on **Python 3.10+** (`requires-python = ">=3.10"`). [CI](https://github.com/TimeCopilot/foundationforecast/actions/workflows/ci.yaml) runs a **preview** job on **3.14** and **3.15** (`continue-on-error: true`) to track dependency wheels and the cross-model test suite as CPython and PyTorch move forward.

On **3.14+**, these optional dependencies are **not** installed (upstream caps): `granite-tsfm`, `tfc-t0`, and `timecopilot-uni2ts`. Wrappers gated to **3.11–3.13** (`FlowState`, `PatchTST-FM`, `T0`) and **&lt; 3.13** (`TabPFN`, `Sundial`) stay unavailable by design. Models that typically remain available on 3.14+ include **Chronos**, **TimesFM**, **Timestral**, **TiRex**, **TimeGPT**, **Toto**, and **Tafsut**, subject to `torch` / `transformers` support on that interpreter.

When preview CI is green, we can promote 3.14 (and later 3.15) into the required **test** matrix.

## Running Tests

```bash
uv run pytest
```

Docs tests (markdown and docstring examples):

```bash
uv run pytest -m docs
```

## Documentation Changes

The docs are built with [Zensical](https://zensical.org), which requires Python 3.11 or newer (the library itself supports 3.10, but the docs tooling is skipped there). Serve them locally with live reload:

```bash
uv run --group docs zensical serve
```

Build docs (strict, as in CI):

```bash
uv run --group docs zensical build
```

Zensical does not render Jupyter notebooks yet, so example pages under `docs/examples/` show up as `.ipynb` downloads in this preview. The deployed site converts them to pages at build time; to preview exactly what will be deployed (converted notebooks and the shared header), run `uv run tc-docs preview --local foundationforecast=../foundationforecast` from a checkout of [TimeCopilot/docs](https://github.com/TimeCopilot/docs).

### Documentation Notes

- Pull requests are tested to ensure documentation builds successfully.
- All documentation files should use **kebab-case** (e.g., `model-hub.md`).

### Adding Models

When adding a new foundation model:

1. Implement the model under `foundationforecast/models/`.
2. Add it to `foundationforecast/models/__init__.py` and version-gated exports in `foundationforecast/__init__.py`.
3. Add the class to `docs/api/models/foundation/models.md`.
4. Add an entry to `docs/model-hub.md`.
5. Consider a family notebook under `docs/examples/`.

## Forked Dependencies

**foundationforecast** uses forked Python packages maintained under custom names on PyPI:

- **chronos-forecasting** — [`timecopilot-chronos-forecasting`](https://pypi.org/project/timecopilot-chronos-forecasting/)
- **timesfm** — [`timecopilot-timesfm`](https://pypi.org/project/timecopilot-timesfm/)
- **tirex** — [`timecopilot-tirex`](https://pypi.org/project/timecopilot-tirex/)
- **toto** — [`timecopilot-toto`](https://pypi.org/project/timecopilot-toto/)
- **uni2ts** — [`timecopilot-uni2ts`](https://pypi.org/project/timecopilot-uni2ts/)
- **timestral** — [`timecopilot-timestral`](https://pypi.org/project/timecopilot-timestral/)

FlowState and PatchTST-FM use upstream [`granite-tsfm`](https://pypi.org/project/granite-tsfm/)
(`>=0.3.9`; PatchTST-FM r2 requires overlapping-patch support). The former
`timecopilot-granite-tsfm` fork is no longer a dependency.

See the [TimeCopilot contributing guide](https://timecopilot.dev/community/contributing/) for upstream fork links.
