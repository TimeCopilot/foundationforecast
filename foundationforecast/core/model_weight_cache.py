"""LRU cache for loaded model weights in GPU/CPU memory."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_DEFAULT_MAX_CACHED_MODELS = 1
_model_weight_cache: ModelWeightCache | None = None


class ModelWeightCache:
    """Keep at most ``max_cached_models`` loaded models; evict least-recently-used."""

    def __init__(self, *, max_cached_models: int = _DEFAULT_MAX_CACHED_MODELS) -> None:
        if max_cached_models < 0:
            raise ValueError("max_cached_models must be >= 0.")
        self._max_cached_models = max_cached_models
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @property
    def max_cached_models(self) -> int:
        return self._max_cached_models

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        if self._max_cached_models == 0:
            return loader()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        model = loader()
        self._cache[key] = model
        while len(self._cache) > self._max_cached_models:
            _, evicted = self._cache.popitem(last=False)
            release_model(evicted)
        return model

    def clear(self, key: str | None = None) -> None:
        if key is None:
            for model in self._cache.values():
                release_model(model)
            self._cache.clear()
            return
        model = self._cache.pop(key, None)
        if model is not None:
            release_model(model)

    def clear_prefix(self, prefix: str) -> None:
        keys = [key for key in self._cache if key.startswith(prefix)]
        for key in keys:
            self.clear(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


def release_model(model: Any) -> None:
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def get_model_weight_cache() -> ModelWeightCache:
    global _model_weight_cache
    if _model_weight_cache is None:
        _model_weight_cache = ModelWeightCache()
    return _model_weight_cache


def set_max_cached_models(max_cached_models: int) -> None:
    """Configure the process-wide LRU weight cache.

    Args:
        max_cached_models: Maximum number of loaded models to keep in memory.
            Defaults to ``1``. Use ``0`` to disable caching entirely.
    """
    global _model_weight_cache
    previous = _model_weight_cache
    _model_weight_cache = ModelWeightCache(max_cached_models=max_cached_models)
    if previous is not None:
        _model_weight_cache._cache = previous._cache.copy()
        while len(_model_weight_cache._cache) > max_cached_models:
            _, evicted = _model_weight_cache._cache.popitem(last=False)
            release_model(evicted)


def reset_model_weight_cache() -> None:
    """Clear and reset the process-wide cache (for tests)."""
    global _model_weight_cache
    if _model_weight_cache is not None:
        _model_weight_cache.clear()
    _model_weight_cache = None
