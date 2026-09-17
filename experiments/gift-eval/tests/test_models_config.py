from src.eval.jobs import load_models_config


def test_t0_beta_registered_in_models_yaml():
    models = load_models_config()
    entry = models["theforecastingcompany--t0-beta"]
    assert entry["class"] == "foundationforecast.models.t0.T0"
    assert entry["reference_slug"] is None
    assert entry["kwargs"]["repo_id"] == "theforecastingcompany/t0-beta"
    assert entry["kwargs"]["alias"] == "t0-beta"
