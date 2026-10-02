from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

FEV_BENCH_RESULTS_BASE = (
    "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/results"
)

REPLICATION_METRIC_COL = "test_error"
REPLICATION_ATOL = 1e-2
REPLICATION_RTOL = 2.5e-2


@lru_cache
def load_reference_results(
    reference_csv: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    cache_root = cache_dir or Path(".pytest_cache") / "fev_bench" / "references"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / reference_csv
    if not cache_file.exists():
        url = f"{FEV_BENCH_RESULTS_BASE}/{reference_csv}"
        df = pd.read_csv(url)
        df.to_csv(cache_file, index=False)
    return pd.read_csv(cache_file)
