from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_RESULTS_ROOT = Path("results")


@dataclass(frozen=True)
class Job:
    model_key: str
    dataset_name: str
    term: str


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


@lru_cache
def load_models_config() -> dict:
    return _load_yaml(CONFIGS_DIR / "models.yaml")["models"]


def load_ci_subset() -> list[Job]:
    raw = _load_yaml(CONFIGS_DIR / "ci_subset.yaml")["jobs"]
    return [Job(**job) for job in raw]


def load_model_matrix() -> list[Job]:
    models = load_models_config()
    return [
        Job(model_key=model_key, dataset_name=dataset_name, term=term)
        for model_key in models
        for dataset_name, term in DATASETS_WITH_TERMS
    ]


def job_output_dir(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return root / job.model_key / job.dataset_name / job.term


def result_csv(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return job_output_dir(job, root) / "all_results.csv"


def timing_json(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return job_output_dir(job, root) / "timing.json"


def ci_output_root() -> Path:
    return DEFAULT_RESULTS_ROOT / "ci"


def jobs_missing_timing(jobs: list[Job], output_root: Path) -> list[Job]:
    missing: list[Job] = []
    for job in jobs:
        has_result = result_csv(job, output_root).exists()
        has_timing = timing_json(job, output_root).exists()
        if has_result and not has_timing:
            missing.append(job)
    return missing
