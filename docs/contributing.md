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

## Running Tests

```bash
uv run pytest
```

Docs tests (markdown and docstring examples):

```bash
uv run pytest -m docs
```

## Documentation Changes

Serve docs locally:

```bash
uv run --group docs mkdocs serve
```

Build docs:

```bash
uv run --group docs mkdocs build
```

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
- **granite-tsfm** — [`timecopilot-granite-tsfm`](https://pypi.org/project/timecopilot-granite-tsfm/)
- **timesfm** — [`timecopilot-timesfm`](https://pypi.org/project/timecopilot-timesfm/)
- **tirex** — [`timecopilot-tirex`](https://pypi.org/project/timecopilot-tirex/)
- **toto** — [`timecopilot-toto`](https://pypi.org/project/timecopilot-toto/)
- **uni2ts** — [`timecopilot-uni2ts`](https://pypi.org/project/timecopilot-uni2ts/)

See the [TimeCopilot contributing guide](https://timecopilot.dev/community/contributing/) for upstream fork links.
