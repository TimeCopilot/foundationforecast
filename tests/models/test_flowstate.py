import sys

import numpy as np
import pandas as pd
import pytest

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    pytest.skip(
        "FlowState requires Python >= 3.11 and < 3.14",
        allow_module_level=True,
    )

from foundationforecast.models.flowstate import FlowState


def test_flowstate_forecast_single_uid():
    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({"unique_id": "u1", "ds": ds, "y": np.arange(20)})

    fcst = FlowState().forecast(df=df, h=1, freq="W")

    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
