from pathlib import Path

import pytest
from src.eval.jobs import load_ci_subset, result_csv
from src.verify.verify import ReplicationSkip, verify_job


def _job_id(job) -> str:
    return f"{job.model_key}-{job.dataset_name}-{job.term}"


@pytest.mark.parametrize("job", load_ci_subset(), ids=_job_id)
def test_replication(job, ci_results_root: Path) -> None:
    csv_path = result_csv(job, ci_results_root)
    if not csv_path.exists():
        pytest.skip(f"Missing CI result: {csv_path}. Run the CI eval step first.")

    try:
        verify_job(job, ci_results_root)
    except ReplicationSkip as exc:
        pytest.skip(str(exc))
