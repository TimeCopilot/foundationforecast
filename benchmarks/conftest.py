"""Shared, deterministic data fixtures for the CodSpeed benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from utilsforecast.data import generate_series


def make_panel(
    n_series: int,
    length: int,
    freq: str = "D",
) -> pd.DataFrame:
    """Generate a deterministic panel with `n_series` series of `length` rows.

    `generate_series` is seeded by default, so the data is stable across runs.
    """
    df = generate_series(
        n_series=n_series,
        freq=freq,
        min_length=length,
        max_length=length,
    )
    df["unique_id"] = df["unique_id"].astype(str)
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def small_panel() -> pd.DataFrame:
    """20 daily series of 200 observations."""
    return make_panel(n_series=20, length=200)


@pytest.fixture(scope="session")
def large_panel() -> pd.DataFrame:
    """200 daily series of 500 observations."""
    return make_panel(n_series=200, length=500)


@pytest.fixture(scope="session")
def unsorted_panel(large_panel: pd.DataFrame) -> pd.DataFrame:
    """A shuffled copy of `large_panel`, forcing the sorting code paths."""
    rng = np.random.default_rng(42)
    idxs = rng.permutation(len(large_panel))
    return large_panel.iloc[idxs].reset_index(drop=True)


@pytest.fixture(scope="session")
def string_ds_panel(large_panel: pd.DataFrame) -> pd.DataFrame:
    """`large_panel` with `ds` stored as ISO-8601 strings."""
    df = large_panel.copy()
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    return df
