import sys

import pytest
from utilsforecast.data import generate_series

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        sys.version_info < (3, 11),
        reason="TiRex requires Python >= 3.11",
    ),
]


def test_is_tirex2_dispatch():
    from foundationforecast.models.tirex import TiRex

    assert not TiRex(repo_id="NX-AI/TiRex")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2")._is_tirex2()


def test_tirex2_forecast_smoke(mocker):
    from foundationforecast.models.tirex import TiRex

    mocker.patch(
        "foundationforecast.models.tirex.load_model",
        return_value=mocker.Mock(
            forecast=mocker.Mock(return_value=(mocker.Mock(), None))
        ),
    )
    df = generate_series(n_series=1, freq="D", min_length=10, max_length=10)
    model = TiRex(repo_id="NX-AI/TiRex-2")
    fcst = model.forecast(df=df, h=2, freq="D")
    assert len(fcst) == 2
