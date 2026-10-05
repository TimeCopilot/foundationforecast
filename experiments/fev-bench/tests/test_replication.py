from pathlib import Path

import pytest
from src.eval.jobs import load_ci_subset, summary_json
from src.verify.verify import ReplicationSkip, verify_job


def _job_id(job) -> str:
    return f"{job.model_key}-{job.task_name}"


@pytest.mark.parametrize("job", load_ci_subset(), ids=_job_id)
def test_replication(job, ci_results_root: Path) -> None:
    summary_path = summary_json(job, ci_results_root)
    if not summary_path.exists():
        pytest.skip(f"Missing CI result: {summary_path}. Run the CI eval step first.")

    try:
        verify_job(job, ci_results_root)
    except ReplicationSkip as exc:
        pytest.skip(str(exc))
