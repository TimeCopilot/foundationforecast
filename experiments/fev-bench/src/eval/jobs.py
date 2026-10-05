from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fev
import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_RESULTS_ROOT = Path("results")


@dataclass(frozen=True)
class Job:
    model_key: str
    task_name: str


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def is_known_dynamic_only(task: fev.Task) -> bool:
    return (
        bool(task.known_dynamic_columns)
        and not task.past_dynamic_columns
        and not task.static_columns
    )


@lru_cache
def load_benchmark_config() -> dict:
    return _load_yaml(CONFIGS_DIR / "benchmark.yaml")


@lru_cache
def load_benchmark() -> fev.Benchmark:
    url = load_benchmark_config()["tasks_url"]
    return fev.Benchmark.from_yaml(url)


@lru_cache
def load_known_only_task_names() -> tuple[str, ...]:
    raw = _load_yaml(CONFIGS_DIR / "known_only_tasks.yaml")["task_names"]
    return tuple(raw)


def load_covariate_tasks(benchmark: fev.Benchmark | None = None) -> list[fev.Task]:
    bench = benchmark or load_benchmark()
    tasks = [t for t in bench.tasks if is_known_dynamic_only(t)]
    expected = set(load_known_only_task_names())
    actual = {t.task_name for t in tasks}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "known_only_tasks.yaml out of sync with fev-bench filter. "
            f"missing={missing} extra={extra}"
        )
    return tasks


def load_task(task_name: str) -> fev.Task:
    for task in load_covariate_tasks():
        if task.task_name == task_name:
            return task
    raise KeyError(
        f"Unknown or non-known-only task {task_name!r}. "
        f"See configs/known_only_tasks.yaml"
    )


@lru_cache
def load_models_config() -> dict:
    return _load_yaml(CONFIGS_DIR / "models.yaml")["models"]


def load_ci_subset() -> list[Job]:
    raw = _load_yaml(CONFIGS_DIR / "ci_subset.yaml")["jobs"]
    return [Job(**job) for job in raw]


def load_covariate_task_matrix() -> list[Job]:
    models = load_models_config()
    tasks = load_covariate_tasks()
    return [
        Job(model_key=model_key, task_name=task.task_name)
        for model_key in models
        for task in tasks
    ]


def job_output_dir(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return root / job.model_key / job.task_name


def summary_json(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return job_output_dir(job, root) / "summary.json"


def summary_csv(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return job_output_dir(job, root) / "summary.csv"


def timing_json(job: Job, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    return job_output_dir(job, root) / "timing.json"


def ci_output_root() -> Path:
    return DEFAULT_RESULTS_ROOT / "ci"
