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


@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch):
    import sys

    def noop_tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return iter([])
        return iterable

    monkeypatch.setattr("tqdm.tqdm", noop_tqdm)
    for name, mod in sys.modules.items():
        if name.startswith("foundationforecast") and hasattr(mod, "tqdm"):
            monkeypatch.setattr(mod, "tqdm", noop_tqdm)
