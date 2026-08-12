import sys

import pytest


def test_timegpt_import():
    # we are not testing timegpt since we need to make api calls to the timegpt api
    from foundationforecast.models.timegpt import TimeGPT  # noqa: F401


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="TiRex requires Python >= 3.11",
)
def test_tirex_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.tirex import TiRex  # noqa: F401
    assert "requires Python >= 3.11" in str(excinfo.value)


@pytest.mark.skipif(
    (3, 11) <= sys.version_info < (3, 14),
    reason="T0 requires Python >= 3.11 and < 3.14",
)
def test_t0_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.t0 import T0  # noqa: F401
    assert "requires Python >= 3.11 and < 3.14" in str(excinfo.value)


@pytest.mark.skipif(
    (3, 11) <= sys.version_info < (3, 14),
    reason="FlowState requires Python >= 3.11 and < 3.14",
)
def test_flowstate_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.flowstate import FlowState  # noqa: F401
    assert "requires Python >= 3.11 and < 3.14" in str(excinfo.value)


@pytest.mark.skipif(
    (3, 11) <= sys.version_info < (3, 14),
    reason="PatchTSTFM requires Python >= 3.11 and < 3.14",
)
def test_patchtst_fm_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.patchtst_fm import PatchTSTFM  # noqa: F401
    assert "requires Python >= 3.11 and < 3.14" in str(excinfo.value)


@pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="Sundial requires Python < 3.13",
)
def test_sundial_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.sundial import Sundial  # noqa: F401
    assert "requires Python < 3.13" in str(excinfo.value)


@pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="TabPFN requires Python < 3.13",
)
def test_tabpfn_import_fails():
    with pytest.raises(ImportError) as excinfo:
        from foundationforecast.models.tabpfn import TabPFN  # noqa: F401
    assert "requires Python < 3.13" in str(excinfo.value)
