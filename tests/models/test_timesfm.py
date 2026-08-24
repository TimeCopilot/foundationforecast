import pytest

from foundationforecast.models.timesfm import (
    _GIFT_EVAL_TORCH_REPOS,
    TimesFM,
    _TimesFMV1,
    _TimesFMV2_p5,
)

pytestmark = pytest.mark.models

MODEL_PARAMS = [
    _TimesFMV1,
    _TimesFMV2_p5,
]


@pytest.mark.parametrize("repo_id", _GIFT_EVAL_TORCH_REPOS)
def test_timesfm_accepts_gift_eval_repos(repo_id):
    model = TimesFM(repo_id=repo_id)
    assert isinstance(model, _TimesFMV1)
    assert model.repo_id == repo_id


def test_timesfm_accepts_pytorch_repos():
    model = TimesFM(repo_id="google/timesfm-1.0-200m-pytorch")
    assert isinstance(model, _TimesFMV1)


def test_timesfm_rejects_non_pytorch_repo():
    with pytest.raises(ValueError, match="pytorch"):
        TimesFM(repo_id="google/timesfm-2.0-500m")


@pytest.mark.parametrize("model_class", MODEL_PARAMS)
def test_model_raises_OSError_on_failed_load(mocker, model_class):
    """Tests that an OSError is raised on a failed load attempt."""
    module_path = "foundationforecast.models.timesfm"
    mocker.patch(f"{module_path}.os.path.exists", return_value=False)
    mocker.patch(f"{module_path}.repo_exists", return_value=False)

    repo_id = "/this-is-a-fake/google/repo-id"

    model_instance = model_class(
        repo_id=repo_id,
        context_length=64,
        batch_size=32,
        alias="test",
    )
    with (
        pytest.raises(OSError, match="Failed to load model"),
        model_instance._get_predictor(prediction_length=12),
    ):
        pass
