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

=== "plot"

    For notebook-style plotting helpers:

    ```bash
    pip install "foundationforecast[plot]"
    ```

## Python version and model availability

Some models are gated by Python version:

| Model | Python requirement |
|-------|-------------------|
| Chronos, Moirai, TimesFM, Toto, TimeGPT, Sundial, Tafsut | 3.10+ |
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
