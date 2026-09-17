import sys

import pandas as pd
import pytest

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    pytest.skip(
        "FlowState requires Python >= 3.11 and < 3.14",
        allow_module_level=True,
    )

from tests.helpers import generate_series  # noqa: E402
from foundationforecast import FoundationForecast
from foundationforecast.models.flowstate import (  # noqa: E402
    FlowState,
    _normalize_freq_for_scale_factor,
)

pytestmark = pytest.mark.models


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("h", "H"),
        ("6h", "6H"),
        ("15min", "15T"),
        ("MS", "M"),
        ("ME", "M"),
        ("D", "D"),
        ("W-MON", "W-MON"),
    ],
)
def test_normalize_freq_for_scale_factor(freq, expected):
    assert _normalize_freq_for_scale_factor(freq) == expected


def test_flowstate_inferred_hourly_freq():
    df = generate_series(n_series=1, freq="H", min_length=20, max_length=20)
    model = FlowState(batch_size=2)
    fcst = model.forecast(df, h=1)
    assert len(fcst) == 1


def test_flowstate_h1_single_uid():
    df = generate_series(n_series=1, freq="W", min_length=20, max_length=20)

    ff = FoundationForecast(models=[FlowState()])

    # this used to crash before the fix
    fcst = ff.forecast(df=df, h=1, freq="W")

    # basic checks
    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
    assert "ds" in fcst.columns
