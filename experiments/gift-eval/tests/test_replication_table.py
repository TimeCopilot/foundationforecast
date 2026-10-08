from pathlib import Path
from urllib.error import HTTPError

import pytest
from src.eval.jobs import REPLICATION_PENDING_HF_REFERENCE
from src.verify import replication_table
from src.verify.replication_table import (
    REPLICATION_TABLE_COLS,
    build_replication_table,
)


def test_build_replication_table_skips_pending_hf_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending_key = next(iter(REPLICATION_PENDING_HF_REFERENCE))

    def _no_network(slug: str):
        raise AssertionError(f"reference download attempted for {slug}")

    monkeypatch.setattr(replication_table, "load_reference_results", _no_network)
    monkeypatch.setattr(
        "src.verify.verify.load_actual_results", lambda key, root: object()
    )

    table = build_replication_table([pending_key], tmp_path, registry="replication")

    assert table.empty
    assert list(table.columns) == REPLICATION_TABLE_COLS


def test_build_replication_table_skips_missing_remote_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_key = "amazon--chronos-2"

    def _not_found(slug: str):
        raise HTTPError("https://example/results", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(replication_table, "load_reference_results", _not_found)
    monkeypatch.setattr(
        "src.verify.verify.load_actual_results", lambda key, root: object()
    )

    table = build_replication_table([model_key], tmp_path, registry="replication")

    assert table.empty
