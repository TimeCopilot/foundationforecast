import pytest

from foundationforecast.models.timesfm import _TimesFMV1, _TimesFMV2_p5

pytestmark = pytest.mark.models

MODEL_PARAMS = [
    _TimesFMV1,
    _TimesFMV2_p5,
]


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
