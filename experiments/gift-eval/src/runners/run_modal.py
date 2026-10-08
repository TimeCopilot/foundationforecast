import logging
from pathlib import Path

import modal

_MODAL_GIFT_EVAL = "/root/gift-eval"
_MODAL_MONOREPO = "/root/monorepo"


def _resolve_gift_eval_root() -> Path:
    here = Path(__file__).resolve()
    try:
        gift_eval_root = here.parents[2]
        if (gift_eval_root / "pyproject.toml").exists():
            return gift_eval_root
    except IndexError:
        pass
    return Path(_MODAL_GIFT_EVAL)


def _resolve_monorepo_root(gift_eval_root: Path) -> Path:
    candidate = gift_eval_root.parent.parent
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "foundationforecast"
    ).is_dir():
        return candidate
    return Path(_MODAL_MONOREPO)


_GIFT_EVAL_ROOT = _resolve_gift_eval_root()
_REPO_ROOT = _resolve_monorepo_root(_GIFT_EVAL_ROOT)

app = modal.App(name="foundationforecast-gift-eval")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "'timecopilot-gift-eval>=0.3.1' modal pyyaml s3fs typer",
    )
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
    .add_local_dir(
        _REPO_ROOT / "foundationforecast",
        remote_path=f"{_MODAL_MONOREPO}/foundationforecast",
        copy=True,
    )
    .run_commands(
        "uv pip install --system --compile-bytecode -e /root/monorepo",
    )
    .add_local_file(
        _GIFT_EVAL_ROOT / "pyproject.toml",
        remote_path=f"{_MODAL_GIFT_EVAL}/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        _GIFT_EVAL_ROOT / "README.md",
        remote_path=f"{_MODAL_GIFT_EVAL}/README.md",
        copy=True,
    )
    .add_local_dir(
        _GIFT_EVAL_ROOT / "src",
        remote_path=f"{_MODAL_GIFT_EVAL}/src",
        copy=True,
    )
    .add_local_dir(
        _GIFT_EVAL_ROOT / "configs",
        remote_path=f"{_MODAL_GIFT_EVAL}/configs",
        copy=True,
    )
    .workdir(_MODAL_GIFT_EVAL)
    .env({"PYTHONPATH": _MODAL_GIFT_EVAL})
    .run_commands(
        "uv pip install --system --no-deps --compile-bytecode -e .",
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


def replication_s3_prefix(run_id: str) -> str:
    return f"results/replication/{run_id}"


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
    registry: str = "default",
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
        registry=registry,  # type: ignore[arg-type]
    )


def _job_tuples(jobs: list) -> list[tuple[str, str, str]]:
    return [(job.model_key, job.dataset_name, job.term) for job in jobs]


def _dispatch_jobs(
    jobs: list,
    *,
    storage_path: str,
    output_root: str,
    force: bool,
    registry: str = "default",
    max_containers: int | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if not jobs:
        logging.info("No jobs to run")
        return
    args = [
        (*job_tuple, storage_path, output_root, force, registry)
        for job_tuple in _job_tuples(jobs)
    ]
    run_fn = run_gift_eval_modal
    if max_containers is not None:
        run_fn = run_gift_eval_modal.with_options(max_containers=max_containers)
        logging.info("Modal max_containers=%s", max_containers)
    results = list(
        run_fn.starmap(
            args,
            return_exceptions=True,
        )
    )
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        for exc in errors[:10]:
            logging.error("Job failed: %s", exc)
        if len(errors) > 10:
            logging.error("... and %s more failures", len(errors) - 10)
    logging.info(
        "Modal batch finished: ok=%s failed=%s total=%s",
        len(results) - len(errors),
        len(errors),
        len(results),
    )
    if errors and len(errors) == len(results):
        raise RuntimeError(f"All Modal jobs failed ({len(errors)} jobs)")


def run_ci_modal(
    jobs: list,
    *,
    storage_path: str = "/s3-bucket/data/gift-eval",
    output_root: str = f"/s3-bucket/{S3_CI_RESULTS_PREFIX}",
) -> None:
    _dispatch_jobs(jobs, storage_path=storage_path, output_root=output_root, force=True)


def _jobs_from_s3(
    jobs: list,
    *,
    bucket: str,
    prefix: str,
    mode: str,
) -> list:
    from src.eval.jobs import filter_jobs_by_s3_mode

    return filter_jobs_by_s3_mode(
        jobs,
        bucket=bucket,
        prefix=prefix,
        mode=mode,
    )


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


@app.local_entrypoint()
def run_replication_pilot(run_id: str, force: bool = True) -> None:
    from src.eval.jobs import load_replication_pilot_jobs

    prefix = replication_s3_prefix(run_id)
    jobs = load_replication_pilot_jobs()
    logging.info(
        "Replication pilot: %s jobs → s3://%s/%s",
        len(jobs),
        S3_BUCKET,
        prefix,
    )
    _dispatch_jobs(
        jobs,
        storage_path="/s3-bucket/data/gift-eval",
        output_root=f"/s3-bucket/{prefix}",
        force=force,
        registry="replication",
    )


@app.local_entrypoint()
def run_replication_full(
    run_id: str,
    force: bool = False,
    model_key: str = "",
    max_containers: int = 20,
) -> None:
    from src.eval.jobs import load_replication_matrix

    prefix = replication_s3_prefix(run_id)
    jobs = load_replication_matrix()
    if model_key:
        jobs = [job for job in jobs if job.model_key == model_key]
    if force:
        selected = jobs
    else:
        selected = _jobs_from_s3(
            jobs,
            bucket=S3_BUCKET,
            prefix=prefix,
            mode="missing",
        )
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Replication full grid: %s jobs (force=%s, max_containers=%s) → s3://%s/%s",
        len(selected),
        force,
        max_containers,
        S3_BUCKET,
        prefix,
    )
    _dispatch_jobs(
        selected,
        storage_path="/s3-bucket/data/gift-eval",
        output_root=f"/s3-bucket/{prefix}",
        force=force,
        registry="replication",
        max_containers=max_containers,
    )
