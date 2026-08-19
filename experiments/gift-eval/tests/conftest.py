from pathlib import Path

import pytest
from src.eval.jobs import ci_output_root, load_ci_subset


@pytest.fixture(scope="session")
def ci_results_root() -> Path:
    return ci_output_root()


@pytest.fixture(scope="session")
def ci_jobs():
    return load_ci_subset()
