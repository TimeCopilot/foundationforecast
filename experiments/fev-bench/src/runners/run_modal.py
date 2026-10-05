import logging
from pathlib import Path

import modal

_MODAL_MONOREPO = "/root/monorepo"
_MODAL_FEV_BENCH = f"{_MODAL_MONOREPO}/experiments/fev-bench"


def _resolve_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    try:
        fev_root = here.parents[2]
        if (fev_root / "pyproject.toml").exists():
            return fev_root, fev_root.parent.parent
    except IndexError:
        pass
    return Path(_MODAL_FEV_BENCH), Path(_MODAL_MONOREPO)


_FEV_BENCH_ROOT, _REPO_ROOT = _resolve_paths()

app = modal.App(name="foundationforecast-fev-bench")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("uv")
    .add_local_file(
        _REPO_ROOT / "pyproject.toml",
        remote_path=f"{_MODAL_MONOREPO}/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        _REPO_ROOT / "README.md",
        remote_path=f"{_MODAL_MONOREPO}/README.md",
        copy=True,
    )
    .add_local_file(
        _REPO_ROOT / "uv.lock",
        remote_path=f"{_MODAL_MONOREPO}/uv.lock",
        copy=True,
    )
    .add_local_dir(
        _REPO_ROOT / "foundationforecast",
        remote_path=f"{_MODAL_MONOREPO}/foundationforecast",
        copy=True,
    )
    .add_local_dir(
        _FEV_BENCH_ROOT,
        remote_path=_MODAL_FEV_BENCH,
        copy=True,
    )
    .workdir(_MODAL_FEV_BENCH)
    .env({"PYTHONPATH": _MODAL_FEV_BENCH})
    .run_commands(
        "uv pip install --system --compile-bytecode -e .",
    )
)
secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
hf_secret = modal.Secret.from_name(
    "hf-secret",
    required_keys=["HF_TOKEN"],
)
volume = {
    "/s3-bucket": modal.CloudBucketMount(
        bucket_name="foundationforecast-fev-bench",
        secret=secret,
    )
}

S3_BUCKET = "foundationforecast-fev-bench"
S3_RESULTS_PREFIX = "results"
S3_CI_RESULTS_PREFIX = "results/ci"


@app.function(
    image=image,
    volumes=volume,
    secrets=[secret, hf_secret],
    timeout=60 * 60 * 6,
    gpu="A10G",
    cpu=8,
)
def run_fev_eval_modal(
    model_key: str,
    task_name: str,
    output_root: str = "/s3-bucket/results",
    force: bool = False,
    num_proc: int = 8,
) -> None:
    import logging
    from pathlib import Path

    from src.eval.evaluate import run_fev_eval
    from src.eval.jobs import Job, summary_json

    logging.basicConfig(level=logging.INFO)
    job = Job(model_key=model_key, task_name=task_name)
    summary_path = summary_json(job, Path(output_root))
    if not force and summary_path.exists():
        logging.info("Skipping existing result at %s", summary_path)
        return
    run_fev_eval(job, output_root=Path(output_root), num_proc=num_proc)


def _dispatch_jobs(
    jobs: list,
    *,
    output_root: str,
    force: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if not jobs:
        logging.info("No jobs to run")
        return
    args = [(job.model_key, job.task_name, output_root, force, 8) for job in jobs]
    results = list(
        run_fev_eval_modal.starmap(
            args,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise RuntimeError(f"Modal jobs failed: {errors}")


def run_ci_modal(
    jobs: list,
    *,
    output_root: str = f"/s3-bucket/{S3_CI_RESULTS_PREFIX}",
) -> None:
    _dispatch_jobs(jobs, output_root=output_root, force=True)


def _s3_job_paths(
    job,
    *,
    bucket: str,
    prefix: str,
) -> tuple[str, str]:
    base = f"s3://{bucket}/{prefix}/{job.model_key}/{job.task_name}"
    return f"{base}/summary.json", f"{base}/timing.json"


def _job_matches_mode(
    *,
    mode: str,
    has_results: bool,
    has_timing: bool,
) -> bool:
    if mode == "missing":
        return not has_results
    if mode == "missing_timing":
        return has_results and not has_timing
    if mode == "all":
        return True
    raise ValueError(f"Unknown job selection mode: {mode!r}")


def _jobs_from_s3(
    jobs: list,
    *,
    bucket: str,
    prefix: str,
    mode: str,
) -> list:
    import fsspec

    fs = fsspec.filesystem("s3")
    selected = []
    for job in jobs:
        results_path, timing_path = _s3_job_paths(job, bucket=bucket, prefix=prefix)
        has_results = fs.exists(results_path)
        has_timing = fs.exists(timing_path)
        if _job_matches_mode(
            mode=mode,
            has_results=has_results,
            has_timing=has_timing,
        ):
            selected.append(job)
    return selected


@app.local_entrypoint()
def run_ci() -> None:
    from src.eval.jobs import load_ci_subset

    jobs = load_ci_subset()
    run_ci_modal(jobs)


@app.local_entrypoint()
def main(force: bool = False) -> None:
    from src.eval.jobs import load_covariate_task_matrix

    jobs = load_covariate_task_matrix()
    if force:
        selected = jobs
    else:
        selected = _jobs_from_s3(
            jobs,
            bucket=S3_BUCKET,
            prefix=S3_RESULTS_PREFIX,
            mode="missing",
        )
    logging.info("Running %s jobs (force=%s)", len(selected), force)
    _dispatch_jobs(
        selected,
        output_root=f"/s3-bucket/{S3_RESULTS_PREFIX}",
        force=force,
    )
