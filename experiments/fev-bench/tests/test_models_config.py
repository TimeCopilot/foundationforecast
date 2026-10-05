from collections import defaultdict

from src.eval.jobs import (
    load_ci_subset,
    load_covariate_tasks,
    load_known_only_task_names,
    load_models_config,
)

CI_UNIVARIATE_TASK = "epf_pjm"
CI_PANEL_TASK = "entsoe_1H"


def test_known_only_yaml_matches_fev_filter():
    expected = set(load_known_only_task_names())
    actual = {t.task_name for t in load_covariate_tasks()}
    assert actual == expected
    assert len(actual) == 13


def test_ci_subset_tasks_are_known_only():
    known = set(load_known_only_task_names())
    for job in load_ci_subset():
        assert job.task_name in known


def test_ci_subset_one_univariate_and_one_panel_per_model():
    by_model: dict[str, set[str]] = defaultdict(set)
    for job in load_ci_subset():
        by_model[job.model_key].add(job.task_name)

    expected_models = set(load_models_config())
    assert set(by_model) == expected_models
    for model_key, tasks in by_model.items():
        assert tasks == {CI_UNIVARIATE_TASK, CI_PANEL_TASK}, model_key


def test_chronos_2_registered():
    models = load_models_config()
    entry = models["amazon--chronos-2"]
    assert entry["class"] == "foundationforecast.models.chronos.Chronos"
    assert entry["reference_csv"] == "chronos-2.csv"
    assert entry["kwargs"]["repo_id"] == "amazon/chronos-2"


def test_timesfm_3_registered_with_leaderboard_reference():
    models = load_models_config()
    entry = models["google--timesfm-3.0-pytorch"]
    assert entry["class"] == "foundationforecast.models.timesfm.TimesFM"
    assert entry["reference_csv"] == "timesfm-3.csv"
    assert entry["kwargs"]["repo_id"] == "google/timesfm-3.0-pytorch"
    assert entry["kwargs"]["alias"] == "TimesFM-3"


def test_t0_beta_registered_with_leaderboard_reference():
    models = load_models_config()
    entry = models["theforecastingcompany--t0-beta"]
    assert entry["class"] == "foundationforecast.models.t0.T0"
    assert entry["reference_csv"] == "t0-beta.csv"
    assert entry["kwargs"]["repo_id"] == "theforecastingcompany/t0-beta"


def test_tirex_2_registered_with_leaderboard_reference():
    models = load_models_config()
    entry = models["NX-AI--TiRex-2"]
    assert entry["class"] == "foundationforecast.models.tirex.TiRex"
    assert entry["reference_csv"] == "tirex-2.csv"
    assert entry["kwargs"]["repo_id"] == "NX-AI/TiRex-2"
