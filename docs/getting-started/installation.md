# Installation

**foundationforecast** can be installed from PyPI:

=== "pip"

    ```bash
    pip install foundationforecast
    ```

=== "uv"

    ```bash
    uv add foundationforecast
    ```

Requires **Python 3.10 or later**.

## Optional extras

=== "ci-proto"

    For [Continuous Intelligence ci-proto](../model-hub.md):

    === "pip"

        ```bash
        pip install "foundationforecast[ci-proto]"
        ```

    === "uv"

        ```bash
        uv add "foundationforecast[ci-proto]"
        ```

=== "plot"

    For notebook-style plotting helpers:

    === "pip"

        ```bash
        pip install "foundationforecast[plot]"
        ```

    === "uv"

        ```bash
        uv add "foundationforecast[plot]"
        ```

=== "prophet"

    For the [Prophet adapter](../prophet-adapter.md):

    === "pip"

        ```bash
        pip install "foundationforecast[prophet]"
        ```

    === "uv"

        ```bash
        uv add "foundationforecast[prophet]"
        ```

## Python version and model availability

Some models are gated by Python version:

| Model | Python requirement |
|-------|-------------------|
| Chronos, Kumo, Moirai, TimesFM, Toto, TimeGPT, Sundial, Tafsut, Tabby | 3.10+ |
| CIProto | 3.10+; install [`foundationforecast[ci-proto]`](#optional-extras) |
| TiRex / TiRex-2 | 3.11+ |
| T0, FlowState, PatchTST-FM | 3.11 – 3.13 |
| TabPFN | 3.10 – 3.12 |

## API keys

- **TimeGPT** requires a [Nixtla](https://nixtla.io/) API key: `export NIXTLA_API_KEY="..."`

## Local development

```bash
git clone git@github.com:<your-username>/foundationforecast.git
cd foundationforecast
uv sync --group dev --group docs
pre-commit install --install-hooks
```

!!! tip

    If you are new to `uv`, see the [uv getting started guide](https://docs.astral.sh/uv/getting-started/).