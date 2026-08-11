import pytest


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
