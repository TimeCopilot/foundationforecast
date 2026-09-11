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
    if not reuse_loaded_model or cache_key is None:
        model = loader()
        try:
            yield model
        finally:
            release_model(model)
    else:
        yield get_model_weight_cache().get_or_load(cache_key, loader)


def clear_model_cache_for_prefix(prefix: str | None) -> None:
    if prefix is not None:
        get_model_weight_cache().clear_prefix(prefix)
