from src.eval.jobs import (
    load_ci_subset,
    load_covariate_tasks,
    load_known_only_task_names,
    load_models_config,
)


def test_known_only_yaml_matches_fev_filter():
    expected = set(load_known_only_task_names())
    actual = {t.task_name for t in load_covariate_tasks()}
    assert actual == expected
    assert len(actual) == 13


def test_ci_subset_tasks_are_known_only():
    known = set(load_known_only_task_names())
    for job in load_ci_subset():
        assert job.task_name in known


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
