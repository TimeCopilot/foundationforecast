import logging

import modal

app = modal.App(name="foundationforecast-gift-eval")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("uv")
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .add_local_file(".python-version", "/root/.python-version", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("configs", remote_path="/root/configs", copy=True)
    .workdir("/root")
    .env({"PYTHONPATH": "/root"})
    .run_commands("uv pip install . --system --compile-bytecode")
)
secret = modal.Secret.from_name(
    "aws-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
volume = {
    "/s3-bucket": modal.CloudBucketMount(
        bucket_name="foundationforecast-gift-eval",
        secret=secret,
    )
}


@app.function(
    image=image,
    volumes=volume,
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

    from ..eval.evaluate import run_gift_eval
    from ..eval.jobs import Job

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


def run_ci_modal(
    jobs: list,
    *,
    storage_path: str = "/s3-bucket/data/gift-eval",
    output_root: str = "/s3-bucket/results/ci",
) -> None:
    logging.basicConfig(level=logging.INFO)
    args = [
        (*job_tuple, storage_path, output_root, True) for job_tuple in _job_tuples(jobs)
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
        raise RuntimeError(f"Modal CI jobs failed: {errors}")


@app.local_entrypoint()
def run_ci() -> None:
    from src.eval.jobs import load_ci_subset

    logging.basicConfig(level=logging.INFO)
    jobs = load_ci_subset()
    run_ci_modal(jobs)


@app.local_entrypoint()
def main() -> None:
    import fsspec

    from src.eval.jobs import load_model_matrix

    logging.basicConfig(level=logging.INFO)
    fs = fsspec.filesystem("s3")
    bucket = "foundationforecast-gift-eval"
    missing_jobs = [
        job
        for job in load_model_matrix()
        if not fs.exists(
            f"s3://{bucket}/results/{job.model_key}/{job.dataset_name}/"
            f"{job.term}/all_results.csv"
        )
    ]
    logging.info("Running %s missing jobs", len(missing_jobs))
    args = [(job.model_key, job.dataset_name, job.term) for job in missing_jobs]
    results = list(
        run_gift_eval_modal.starmap(
            args,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    errors = [result for result in results if isinstance(result, Exception)]
    logging.info("errors: %s", errors)
