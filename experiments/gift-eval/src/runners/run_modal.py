import logging
from pathlib import Path

import modal

_MODAL_MONOREPO = "/root/monorepo"
_MODAL_GIFT_EVAL = f"{_MODAL_MONOREPO}/experiments/gift-eval"


def _resolve_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    try:
        gift_eval_root = here.parents[2]
        if (gift_eval_root / "pyproject.toml").exists():
            return gift_eval_root, gift_eval_root.parent.parent
    except IndexError:
        pass
    return Path(_MODAL_GIFT_EVAL), Path(_MODAL_MONOREPO)


_GIFT_EVAL_ROOT, _REPO_ROOT = _resolve_paths()

app = modal.App(name="foundationforecast-gift-eval")
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
        _GIFT_EVAL_ROOT,
        remote_path=_MODAL_GIFT_EVAL,
        copy=True,
    )
    .workdir(_MODAL_GIFT_EVAL)
    .env({"PYTHONPATH": _MODAL_GIFT_EVAL})
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
        bucket_name="foundationforecast-gift-eval",
        secret=secret,
    )
}

S3_BUCKET = "foundationforecast-gift-eval"
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
def run_gift_eval_modal(
    model_key: str,
    dataset_name: str,
    term: str,
    storage_path: str = "/s3-bucket/data/gift-eval",
    output_root: str = "/s3-bucket/results",
    force: bool = False,
) -> None:
    import logging
    from pathlib import Path

    from src.eval.evaluate import run_gift_eval
    from src.eval.jobs import Job

    logging.basicConfig(level=logging.INFO)
    job = Job(model_key=model_key, dataset_name=dataset_name, term=term)
    output_path = (
        Path(output_root) / model_key / dataset_name / term / "all_results.csv"
    )
    if not force and output_path.exists():
        logging.info("Skipping existing result at %s", output_path)
        return
    run_gift_eval(
        job,
        storage_path=storage_path,
        output_root=Path(output_root),
        overwrite_results=force,
    )


def _job_tuples(jobs: list) -> list[tuple[str, str, str]]:
    return [(job.model_key, job.dataset_name, job.term) for job in jobs]


def _dispatch_jobs(
    jobs: list,
    *,
    storage_path: str,
    output_root: str,
    force: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if not jobs:
        logging.info("No jobs to run")
        return
    args = [
        (*job_tuple, storage_path, output_root, force)
        for job_tuple in _job_tuples(jobs)
    ]
    results = list(
        run_gift_eval_modal.starmap(
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
    storage_path: str = "/s3-bucket/data/gift-eval",
    output_root: str = f"/s3-bucket/{S3_CI_RESULTS_PREFIX}",
) -> None:
    _dispatch_jobs(jobs, storage_path=storage_path, output_root=output_root, force=True)


def _s3_job_paths(
    job,
    *,
    bucket: str,
    prefix: str,
) -> tuple[str, str]:
    base = f"s3://{bucket}/{prefix}/{job.model_key}/{job.dataset_name}/{job.term}"
    return f"{base}/all_results.csv", f"{base}/timing.json"


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
    from src.eval.jobs import load_model_matrix

    jobs = load_model_matrix()
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
        storage_path="/s3-bucket/data/gift-eval",
        output_root=f"/s3-bucket/{S3_RESULTS_PREFIX}",
        force=force,
    )


@app.local_entrypoint()
def run_missing_timing() -> None:
    from src.eval.jobs import load_model_matrix

    jobs = load_model_matrix()
    selected = _jobs_from_s3(
        jobs,
        bucket=S3_BUCKET,
        prefix=S3_RESULTS_PREFIX,
        mode="missing_timing",
    )
    logging.info("Backfilling timing for %s jobs", len(selected))
    _dispatch_jobs(
        selected,
        storage_path="/s3-bucket/data/gift-eval",
        output_root=f"/s3-bucket/{S3_RESULTS_PREFIX}",
        force=True,
    )
