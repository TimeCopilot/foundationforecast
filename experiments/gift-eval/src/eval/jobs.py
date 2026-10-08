from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from timecopilot_gift_eval.utils import DATASETS_WITH_TERMS

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
REPLICATION_CONFIGS_DIR = CONFIGS_DIR / "replication"
DEFAULT_RESULTS_ROOT = Path("results")
REPLICATION_PILOT_DATASET = "m4_weekly"
REPLICATION_PILOT_TERM = "short"

# Families in the replication grid before GIFT-Eval HF results/{slug} exists.
REPLICATION_PENDING_HF_REFERENCE = frozenset({"autogluon--chronos-2-small"})


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


@lru_cache
def load_replication_models_config() -> dict:
    return _load_yaml(REPLICATION_CONFIGS_DIR / "models.yaml")["models"]


def load_models_config_for_registry(registry: str) -> dict:
    if registry == "default":
        return load_models_config()
    if registry == "replication":
        return load_replication_models_config()
    raise ValueError(f"Unknown registry {registry!r}")


@lru_cache
def load_replication_families() -> tuple[str, ...]:
    raw = _load_yaml(REPLICATION_CONFIGS_DIR / "families.yaml")["model_keys"]
    return tuple(raw)


def load_replication_pilot_jobs() -> list[Job]:
    return [
        Job(
            model_key=model_key,
            dataset_name=REPLICATION_PILOT_DATASET,
            term=REPLICATION_PILOT_TERM,
        )
        for model_key in load_replication_families()
    ]


def load_replication_matrix() -> list[Job]:
    return [
        Job(model_key=model_key, dataset_name=dataset_name, term=term)
        for model_key in load_replication_families()
        for dataset_name, term in DATASETS_WITH_TERMS
    ]


def replication_output_root(run_id: str) -> Path:
    return DEFAULT_RESULTS_ROOT / "replication" / run_id


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


def job_key(job: Job) -> tuple[str, str, str]:
    return (job.model_key, job.dataset_name, job.term)


def parse_job_dir_suffix(relative_dir: str) -> tuple[str, str, str] | None:
    """Parse ``model_key/dataset/term`` (dataset may contain ``/``)."""
    parts = relative_dir.strip("/").split("/")
    if len(parts) < 3:
        return None
    term = parts[-1]
    model_key = parts[0]
    dataset_name = "/".join(parts[1:-1])
    return (model_key, dataset_name, term)


def index_s3_job_artifacts(
    *,
    bucket: str,
    prefix: str,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """List ``all_results.csv`` / ``timing.json`` under prefix in one S3 walk."""
    import fsspec

    fs = fsspec.filesystem("s3")
    base = f"{bucket}/{prefix.strip('/')}/"
    has_results: set[tuple[str, str, str]] = set()
    has_timing: set[tuple[str, str, str]] = set()
    try:
        paths = fs.find(base)
    except FileNotFoundError:
        return has_results, has_timing
    for path in paths:
        if not path.startswith(base):
            continue
        rel = path[len(base) :]
        if rel.endswith("/all_results.csv"):
            key = parse_job_dir_suffix(rel[: -len("/all_results.csv")])
            if key is not None:
                has_results.add(key)
        elif rel.endswith("/timing.json"):
            key = parse_job_dir_suffix(rel[: -len("/timing.json")])
            if key is not None:
                has_timing.add(key)
    return has_results, has_timing


def filter_jobs_by_s3_mode(
    jobs: list[Job],
    *,
    bucket: str,
    prefix: str,
    mode: str,
) -> list[Job]:
    if mode == "all":
        return list(jobs)
    if mode not in ("missing", "missing_timing"):
        raise ValueError(f"Unknown job selection mode: {mode!r}")
    has_results, has_timing = index_s3_job_artifacts(bucket=bucket, prefix=prefix)
    selected: list[Job] = []
    for job in jobs:
        key = job_key(job)
        has_r = key in has_results
        has_t = key in has_timing
        if mode == "missing":
            if not has_r:
                selected.append(job)
        elif has_r and not has_t:
            selected.append(job)
    return selected


def jobs_missing_timing(jobs: list[Job], output_root: Path) -> list[Job]:
    missing: list[Job] = []
    for job in jobs:
        has_result = result_csv(job, output_root).exists()
        has_timing = timing_json(job, output_root).exists()
        if has_result and not has_timing:
            missing.append(job)
    return missing
