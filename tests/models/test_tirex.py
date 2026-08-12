import sys

import pytest

if sys.version_info < (3, 11):
    pytest.skip(
        "TiRex requires Python >= 3.11",
        allow_module_level=True,
    )

from tests.helpers import generate_series  # noqa: E402
from foundationforecast.models.tirex import TiRex  # noqa: E402

pytestmark = pytest.mark.models


def test_is_tirex2_dispatch():
    assert not TiRex(repo_id="NX-AI/TiRex")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2/")._is_tirex2()


def test_tirex2_forecast():
    df = generate_series(2, freq="D", min_length=50, max_length=50)
    model = TiRex(repo_id="NX-AI/TiRex-2", alias="TiRex-2", batch_size=2)
    fcst = model.forecast(df, h=3, freq="D")
    assert fcst.shape == (6, 3)
    assert "TiRex-2" in fcst.columns
