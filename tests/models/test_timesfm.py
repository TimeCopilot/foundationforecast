from contextlib import contextmanager
from io import StringIO

import pandas as pd
import pytest

from tests.helpers import generate_series
from foundationforecast.models.timesfm import (
    _GIFT_EVAL_LEGACY_REPOS,
    TimesFM,
    _TimesFMV1,
    _TimesFMV2_p5,
    _TimesFMV3,
)

pytestmark = pytest.mark.models

MODEL_PARAMS = [
    _TimesFMV1,
    _TimesFMV2_p5,
    _TimesFMV3,
]


@pytest.mark.parametrize("repo_id", _GIFT_EVAL_LEGACY_REPOS)
def test_timesfm_accepts_gift_eval_repos(repo_id):
    model = TimesFM(repo_id=repo_id)
    assert isinstance(model, _TimesFMV1)
    assert model.repo_id == repo_id


def test_timesfm_accepts_pytorch_repos():
    model = TimesFM(repo_id="google/timesfm-1.0-200m-pytorch")
    assert isinstance(model, _TimesFMV1)


def test_timesfm_routes_3_0_repo():
    model = TimesFM(repo_id="google/timesfm-3.0-pytorch")
    assert isinstance(model, _TimesFMV3)
    assert model.repo_id == "google/timesfm-3.0-pytorch"


def test_timesfm_rejects_non_pytorch_repo():
    with pytest.raises(ValueError, match="JAX backends are not supported"):
        TimesFM(repo_id="google/timesfm-2.0-500m")


def test_timesfm_v1_forecast_converts_string_ds_to_datetime(mocker):
    """JSON roundtrips leave ds as object dtype; forecast must still work."""
    df = generate_series(n_series=1, freq="D", min_length=14, max_length=14)
    df = pd.read_json(
        StringIO(df.to_json(orient="split", date_format="iso")),
        orient="split",
    )
    assert df["ds"].dtype == object

    model = _TimesFMV1(
        repo_id="google/timesfm-2.0-500m-pytorch",
        context_length=512,
        batch_size=32,
        alias="TimesFM",
    )
    last_time = pd.to_datetime(df.groupby("unique_id")["ds"].tail(1).iloc[0])
    expected_fcst = pd.DataFrame(
        {
            "unique_id": [df["unique_id"].iloc[0]] * 2,
            "ds": pd.date_range(last_time, periods=3, freq="D")[1:],
            "TimesFM": [1.0, 2.0],
        }
    )
    mock_predictor = mocker.Mock()
    mock_predictor.forecast_on_df.return_value = expected_fcst

    @contextmanager
    def fake_predictor(*_args, **_kwargs):
        yield mock_predictor

    mocker.patch.object(model, "_get_predictor", side_effect=fake_predictor)

    fcst = model.forecast(df=df, h=2, freq="D")

    assert fcst.equals(expected_fcst)
    inputs = mock_predictor.forecast_on_df.call_args.kwargs["inputs"]
    assert pd.api.types.is_datetime64_any_dtype(inputs["ds"])


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
