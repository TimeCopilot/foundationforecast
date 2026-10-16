from src.eval.jobs import load_models_config


def test_t0_beta_registered_in_models_yaml():
    models = load_models_config()
    entry = models["theforecastingcompany--t0-beta"]
    assert entry["class"] == "foundationforecast.models.t0.T0"
    assert entry["reference_slug"] is None
    assert entry["kwargs"]["repo_id"] == "theforecastingcompany/t0-beta"
    assert entry["kwargs"]["alias"] == "t0-beta"


def test_timestral_209m_registered_with_leaderboard_settings():
    models = load_models_config()
    entry = models["mistralai--Timestral-209M"]
    assert entry["class"] == "foundationforecast.models.timestral.Timestral"
    assert entry["reference_slug"] == "Timestral-209M"
    assert entry["kwargs"]["repo_id"] == "mistralai/Timestral-209M"


def test_granite_patchtst_r2_registered_in_models_yaml():
    models = load_models_config()
    entry = models["ibm-granite--granite-timeseries-patchtst-fm-r2"]
    assert entry["class"] == "foundationforecast.models.patchtst_fm.PatchTSTFM"
    assert entry["reference_slug"] == "Granite-PatchTST-FM-r2"
    assert entry["max_length"] == 8192
    assert entry["kwargs"]["repo_id"] == "ibm-granite/granite-timeseries-patchtst-fm-r2"
    assert entry["kwargs"]["context_length"] == 8192
