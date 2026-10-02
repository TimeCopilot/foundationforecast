"""Build multi-market EPF panel (fev-bench epf_*) for S3 train/test parquets.

Train: fev evaluation window 0 past data for each market.
Test: horizon exog + ground-truth y (24 steps per market).

Exogenous columns are renamed to ex_1, ex_2 (order = task.known_dynamic_columns).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fev
import pandas as pd
from fev.adapters import PandasAdapter

FEV_BENCH_TASKS_URL = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml"
EPF_TASK_PREFIX = "epf_"


def _frame_for_market(
    task: fev.Task, window_index: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = task.get_window(window_index)
    past, future = window.get_input_data()
    past_df, future_df, _ = PandasAdapter.convert_input_data(
        past,
        future,
        target_columns=window.target_columns,
        id_column=window.id_column,
        timestamp_column=window.timestamp_column,
        static_columns=window.static_columns,
    )
    gt_df = PandasAdapter._to_long_df(window.get_ground_truth(), window.id_column)

    exog_cols = list(task.known_dynamic_columns)
    if len(exog_cols) != 2:
        raise ValueError(
            f"{task.task_name}: expected 2 known covariates, got {exog_cols}"
        )

    rename = {
        task.id_column: "unique_id",
        task.timestamp_column: "ds",
        window.target_columns[0]: "y",
        exog_cols[0]: "ex_1",
        exog_cols[1]: "ex_2",
    }

    train = past_df.rename(columns=rename)[list(rename.values())]
    exog_rename = {exog_cols[0]: "ex_1", exog_cols[1]: "ex_2"}
    test_exog = future_df.rename(
        columns={
            task.id_column: "unique_id",
            task.timestamp_column: "ds",
            **exog_rename,
        }
    )[["unique_id", "ds", "ex_1", "ex_2"]]
    test_y = gt_df.rename(columns=rename)[["unique_id", "ds", "y"]]
    test = test_exog.merge(test_y, on=["unique_id", "ds"], how="inner")
    return train, test


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark = fev.Benchmark.from_yaml(FEV_BENCH_TASKS_URL)
    epf_tasks = sorted(
        (t for t in benchmark.tasks if t.task_name.startswith(EPF_TASK_PREFIX)),
        key=lambda t: t.task_name,
    )
    if not epf_tasks:
        raise RuntimeError("No epf_* tasks found in fev-bench")

    trains: list[pd.DataFrame] = []
    tests: list[pd.DataFrame] = []
    for task in epf_tasks:
        train, test = _frame_for_market(task)
        trains.append(train)
        tests.append(test)

    train_all = pd.concat(trains, ignore_index=True)
    test_all = pd.concat(tests, ignore_index=True)
    train_all = train_all.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    test_all = test_all.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return train_all, test_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/electricity_price"),
        help="Directory for train.parquet and test.parquet",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, test = build_panel()
    train_path = args.out_dir / "train.parquet"
    test_path = args.out_dir / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"Wrote {train_path} rows={len(train)} series={train['unique_id'].nunique()}")
    print(f"Wrote {test_path} rows={len(test)} series={test['unique_id'].nunique()}")
    print("Markets:", sorted(train["unique_id"].unique().tolist()))


if __name__ == "__main__":
    main()
