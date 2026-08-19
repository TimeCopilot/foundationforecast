from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_results(
    *,
    bucket: str = "foundationforecast-gift-eval",
    model_key: str,
    output_dir: Path = Path("results"),
) -> Path:
    dfs: list[pd.DataFrame] = []

    for dataset_name, term in DATASETS_WITH_TERMS:
        csv_path = (
            f"s3://{bucket}/results/{model_key}/{dataset_name}/{term}/all_results.csv"
        )
        logger.info("Downloading %s", csv_path)
        try:
            df = pd.read_csv(csv_path, storage_options={"anon": False})
            dfs.append(df)
        except Exception as exc:
            logger.error("Error downloading %s: %s", csv_path, exc)

    if not dfs:
        raise RuntimeError(f"No results downloaded for model_key={model_key!r}")

    consolidated = pd.concat(dfs, ignore_index=True)
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / "all_results.csv"
    consolidated.to_csv(output_path, index=False)
    logger.info("Saved consolidated results to %s", output_path)
    return output_path


if __name__ == "__main__":
    from typing import Annotated

    import typer

    cli = typer.Typer()

    @cli.command()
    def main(
        model_key: Annotated[str, typer.Option(help="Model key to consolidate")],
        bucket: Annotated[
            str,
            typer.Option(help="S3 bucket containing benchmark results"),
        ] = "foundationforecast-gift-eval",
        output_dir: Annotated[
            Path,
            typer.Option(help="Local output directory"),
        ] = Path("results"),
    ) -> None:
        download_results(bucket=bucket, model_key=model_key, output_dir=output_dir)

    cli()
