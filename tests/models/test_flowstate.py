import sys

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    pytest.skip(
        "FlowState requires Python >= 3.11 and < 3.14",
        allow_module_level=True,
    )

from foundationforecast import FoundationForecast
from foundationforecast.models.flowstate import FlowState  # noqa: E402

pytestmark = pytest.mark.models


def test_flowstate_h1_single_uid():
    # create simple weekly data for one unique_id
    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({"unique_id": "u1", "ds": ds, "y": np.arange(20)})

    ff = FoundationForecast(models=[FlowState()])

    # this used to crash before the fix
    fcst = ff.forecast(df=df, h=1, freq="W")

    # basic checks
    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
    assert "ds" in fcst.columns
