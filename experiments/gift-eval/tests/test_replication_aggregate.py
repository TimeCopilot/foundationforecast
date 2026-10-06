from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src.eval.jobs import Job
from src.verify.reference import CRPS_COL, MASE_COL
from src.verify.replication_aggregate import (
    REPLICATION_AGGREGATE_RTOL,
    leaderboard_aggregate,
    verify_replication_aggregate,
)


def test_leaderboard_aggregate_geomean_normalized() -> None:
    sn = pd.DataFrame(
        {
            "dataset": ["cfg/a", "cfg/b"],
            "sn_mase": [2.0, 0.5],
            "sn_crps": [4.0, 1.0],
        }
    )
    df = pd.DataFrame(
        {
            "dataset": ["cfg/a", "cfg/b"],
            MASE_COL: [1.0, 0.5],
            CRPS_COL: [2.0, 0.5],
        }
    )
    scores = leaderboard_aggregate(df, sn)
    assert scores.n_configs == 2
    assert np.isclose(scores.norm_mase, stats.gmean([0.5, 1.0]))
    assert np.isclose(scores.norm_crps, stats.gmean([0.5, 0.5]))


def test_verify_replication_aggregate_passes_with_outlier_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_key = "test--model"
    datasets = ["cfg/a", "cfg/b"]
    sn = pd.DataFrame(
        {
            "dataset": datasets,
            "sn_mase": [1.0, 1.0],
            "sn_crps": [1.0, 1.0],
        }
    )
    # HF submission: geomean MASE/SN = gmean(0.7, 0.7) = 0.7
    hf_rows = pd.DataFrame(
        {
            "dataset": datasets,
            MASE_COL: [0.7, 0.7],
            CRPS_COL: [0.5, 0.5],
        }
    )
    # Per-job values differ (>2.5% rtol) but geomean stays near 0.7
    rep_rows = pd.DataFrame(
        {
            "dataset": datasets,
            MASE_COL: [0.85, 0.58],
            CRPS_COL: [0.5, 0.5],
        }
    )

    model_root = tmp_path / model_key / "m4_weekly" / "short"
    model_root.mkdir(parents=True)
    rep_rows.to_csv(model_root / "all_results.csv", index=False)

    monkeypatch.setattr(
        "src.verify.replication_aggregate.reference_slug",
        lambda _key, registry="replication": "hf-slug",
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.load_reference_results",
        lambda slug: hf_rows.copy(),
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.seasonal_naive_by_dataset",
        lambda: sn.copy(),
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.load_replication_matrix",
        lambda: [
            Job(model_key=model_key, dataset_name="m4_weekly", term="short"),
        ],
    )

    verify_replication_aggregate(
        model_key,
        tmp_path,
        require_complete=True,
        rtol=REPLICATION_AGGREGATE_RTOL,
    )


def test_verify_replication_aggregate_fails_when_aggregate_drifts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_key = "test--model"
    datasets = ["cfg/a"]
    sn = pd.DataFrame({"dataset": datasets, "sn_mase": [1.0], "sn_crps": [1.0]})
    hf_rows = pd.DataFrame({"dataset": datasets, MASE_COL: [0.5], CRPS_COL: [0.5]})
    rep_rows = pd.DataFrame({"dataset": datasets, MASE_COL: [0.8], CRPS_COL: [0.5]})

    model_root = tmp_path / model_key / "m4_weekly" / "short"
    model_root.mkdir(parents=True)
    rep_rows.to_csv(model_root / "all_results.csv", index=False)

    monkeypatch.setattr(
        "src.verify.replication_aggregate.reference_slug",
        lambda _key, registry="replication": "hf-slug",
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.load_reference_results",
        lambda slug: hf_rows.copy(),
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.seasonal_naive_by_dataset",
        lambda: sn.copy(),
    )
    monkeypatch.setattr(
        "src.verify.replication_aggregate.load_replication_matrix",
        lambda: [
            Job(model_key=model_key, dataset_name="m4_weekly", term="short"),
        ],
    )

    with pytest.raises(AssertionError, match="MASE/Seasonal_Naive"):
        verify_replication_aggregate(
            model_key,
            tmp_path,
            require_complete=True,
            rtol=REPLICATION_AGGREGATE_RTOL,
        )
