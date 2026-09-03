from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.eval.models import reference_slug
from src.verify.reference import CRPS_COL, MASE_COL, load_reference_results

logger = logging.getLogger(__name__)

REPLICATION_TABLE_COLS = [
    "dataset",
    "model",
    "model_key",
    "time_seconds",
    "mase",
    "crps",
    "reported_gift_eval_mase",
    "reported_gift_eval_crps",
    "mase_diff",
    "crps_diff",
]


def load_timing_map(model_key: str, output_root: Path) -> dict[str, float]:
    timings: dict[str, float] = {}
    model_root = output_root / model_key
    if not model_root.exists():
        return timings

    for timing_path in model_root.glob("**/timing.json"):
        results_path = timing_path.parent / "all_results.csv"
        if not results_path.exists():
            continue
        results_df = pd.read_csv(results_path)
        if results_df.empty:
            continue
        dataset = results_df["dataset"].iloc[0]
        with timing_path.open() as f:
            payload = json.load(f)
        timings[dataset] = float(payload["elapsed_seconds"])
    return timings


def build_replication_table(
    model_keys: list[str],
    output_root: Path,
) -> pd.DataFrame:
    from src.verify.verify import load_actual_results

    rows: list[dict] = []
    for model_key in model_keys:
        slug = reference_slug(model_key)
        if slug is None:
            logger.warning("Skipping %s: no reference_slug", model_key)
            continue

        try:
            actual = load_actual_results(model_key, output_root)
            expected = load_reference_results(slug)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", model_key, exc)
            continue

        timing_map = load_timing_map(model_key, output_root)
        expected_by_dataset = expected.set_index("dataset")

        for _, row in actual.iterrows():
            dataset = row["dataset"]
            if dataset not in expected_by_dataset.index:
                continue

            reference = expected_by_dataset.loc[dataset]
            mase = float(row[MASE_COL])
            crps = float(row[CRPS_COL])
            reported_mase = float(reference[MASE_COL])
            reported_crps = float(reference[CRPS_COL])

            rows.append(
                {
                    "dataset": dataset,
                    "model": row["model"],
                    "model_key": model_key,
                    "time_seconds": timing_map.get(dataset),
                    "mase": mase,
                    "crps": crps,
                    "reported_gift_eval_mase": reported_mase,
                    "reported_gift_eval_crps": reported_crps,
                    "mase_diff": mase - reported_mase,
                    "crps_diff": crps - reported_crps,
                }
            )

    if not rows:
        return pd.DataFrame(columns=REPLICATION_TABLE_COLS)

    table = pd.DataFrame(rows)[REPLICATION_TABLE_COLS]
    return table.sort_values(["model_key", "dataset"]).reset_index(drop=True)


def write_replication_table(
    model_keys: list[str],
    output_root: Path,
    table_path: Path,
) -> pd.DataFrame:
    table = build_replication_table(model_keys, output_root)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    logger.info("Wrote replication table (%s rows) to %s", len(table), table_path)
    return table
