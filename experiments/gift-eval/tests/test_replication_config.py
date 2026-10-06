import pytest
from src.eval.jobs import load_replication_families, load_replication_pilot_jobs
from src.eval.models import build_model, reference_slug
from src.verify.reference import load_reference_results


@pytest.mark.parametrize("model_key", load_replication_families())
def test_replication_alias_matches_hf_reference_model_column(model_key: str) -> None:
    slug = reference_slug(model_key, registry="replication")
    assert slug is not None, f"missing reference_slug for {model_key}"
    expected_name = load_reference_results(slug)["model"].iloc[0]
    forecaster = build_model(model_key, registry="replication")
    assert forecaster.alias == expected_name


def test_replication_pilot_job_count() -> None:
    families = load_replication_families()
    pilot = load_replication_pilot_jobs()
    assert len(pilot) == len(families)
    assert {j.model_key for j in pilot} == set(families)


def test_timesfm3_replication_context_length() -> None:
    from src.eval.jobs import load_replication_models_config

    spec = load_replication_models_config()["google--timesfm-3.0-pytorch"]
    assert spec["kwargs"]["context_length"] == 15360


def test_moirai2_replication_context_and_batch() -> None:
    from src.eval.jobs import load_replication_models_config

    spec = load_replication_models_config()["Salesforce--moirai-2.0-R-small"]
    assert spec["kwargs"]["context_length"] == 4000
    assert spec["kwargs"]["batch_size"] == 2048


def test_toto_replication_batch_size() -> None:
    from src.eval.jobs import load_replication_models_config

    spec = load_replication_models_config()["Datadog--Toto-2.0-313m"]
    assert spec["kwargs"]["batch_size"] == 512
