import sys

import pytest


def test_foundation_forecast_import():
    from foundationforecast import FoundationForecast  # noqa: F401


def test_core_exports_import():
    from foundationforecast import (  # noqa: F401
        Chronos,
        Moirai,
        TimeGPT,
        TimesFM,
        Toto,
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="TiRex requires Python >= 3.11",
)
def test_tirex_export_import():
    from foundationforecast import TiRex  # noqa: F401


@pytest.mark.skipif(
    not ((3, 11) <= sys.version_info < (3, 14)),
    reason="FlowState requires Python >= 3.11 and < 3.14",
)
def test_flowstate_export_import():
    from foundationforecast import T0, FlowState, PatchTSTFM  # noqa: F401


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="Sundial and TabPFN require Python < 3.13",
)
def test_sundial_tabpfn_export_import():
    from foundationforecast import Sundial, TabPFN  # noqa: F401
