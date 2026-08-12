import sys

import pytest

from foundationforecast.models.chronos import Chronos
from foundationforecast.models.moirai import Moirai


@pytest.fixture(autouse=True)
def disable_mps_session(monkeypatch):
    try:
        import torch

        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: False, raising=False
        )
        monkeypatch.setattr(
            torch.backends.mps, "is_built", lambda: False, raising=False
        )
    except Exception:
        pass


models = [
    Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos"),
    Moirai(repo_id="Salesforce/moirai-1.0-R-small", alias="Moirai"),
]

if sys.version_info >= (3, 11):
    from foundationforecast.models.tirex import TiRex

    models.append(TiRex())
