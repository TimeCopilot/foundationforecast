"""Helpers for reusing loaded model weights across forecast calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from .model_weight_cache import get_model_weight_cache, release_model

T = TypeVar("T")


@contextmanager
def cached_model_context(
    *,
    reuse_loaded_model: bool,
    cache_key: str | None,
    loader: Callable[[], T],
) -> Iterator[T]:
    cache = get_model_weight_cache()
    use_cache = (
        reuse_loaded_model and cache_key is not None and cache.max_cached_models > 0
    )
    if not use_cache:
        model = loader()
        try:
            yield model
        finally:
            release_model(model)
    else:
        assert cache_key is not None
        yield cache.get_or_load(cache_key, loader)


def clear_model_cache_for_prefix(prefix: str | None) -> None:
    if prefix is not None:
        get_model_weight_cache().clear_prefix(prefix)
