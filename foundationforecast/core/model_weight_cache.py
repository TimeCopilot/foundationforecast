"""LRU cache for loaded model weights in GPU/CPU memory."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

_DEFAULT_MAX_CACHED_MODELS = 1
_model_weight_cache: ModelWeightCache | None = None
_cache_init_lock = threading.Lock()


@dataclass
class _InflightLoad:
    event: threading.Event = field(default_factory=threading.Event)
    model: Any | None = None
    error: BaseException | None = None


class ModelWeightCache:
    """Keep at most ``max_cached_models`` loaded models; evict least-recently-used."""

    def __init__(self, *, max_cached_models: int = _DEFAULT_MAX_CACHED_MODELS) -> None:
        if max_cached_models < 0:
            raise ValueError("max_cached_models must be >= 0.")
        self._max_cached_models = max_cached_models
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._inflight: dict[str, _InflightLoad] = {}
        self._lock = threading.Lock()

    @property
    def max_cached_models(self) -> int:
        return self._max_cached_models

    def _evict_if_needed_locked(self) -> list[Any]:
        evicted: list[Any] = []
        while len(self._cache) >= self._max_cached_models:
            evicted.append(self._cache.popitem(last=False)[1])
        return evicted

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        if self._max_cached_models == 0:
            return loader()

        while True:
            evicted: list[Any] = []
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached

                inflight = self._inflight.get(key)
                if inflight is None:
                    evicted = self._evict_if_needed_locked()
                    inflight = _InflightLoad()
                    self._inflight[key] = inflight
                    is_loader = True
                else:
                    is_loader = False

            for model in evicted:
                release_model(model)

            if is_loader:
                try:
                    loaded = loader()
                except BaseException as exc:
                    with self._lock:
                        inflight = self._inflight.pop(key, inflight)
                        inflight.error = exc
                        inflight.event.set()
                    raise

                with self._lock:
                    inflight = self._inflight.pop(key, inflight)
                    cached = self._cache.get(key)
                    if cached is not None:
                        release_model(loaded)
                        self._cache.move_to_end(key)
                        inflight.model = cached
                        inflight.event.set()
                        return cached
                    extra_evicted = self._evict_if_needed_locked()
                    self._cache[key] = loaded
                    inflight.model = loaded
                    inflight.event.set()
                    result = loaded

                for model in extra_evicted:
                    release_model(model)
                return result

            inflight.event.wait()
            with self._lock:
                if inflight.error is not None:
                    raise inflight.error
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    return cached

    def clear(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                models = list(self._cache.values())
                self._cache.clear()
            else:
                model = self._cache.pop(key, None)
                models = [model] if model is not None else []
        for model in models:
            release_model(model)
        _empty_device_cache()

    def clear_prefix(self, prefix: str) -> None:
        boundary = f"{prefix}:"
        with self._lock:
            keys = [
                key for key in self._cache if key == prefix or key.startswith(boundary)
            ]
        for key in keys:
            self.clear(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache


def _empty_device_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def release_model(model: Any) -> None:
    del model
    _empty_device_cache()


def get_model_weight_cache() -> ModelWeightCache:
    global _model_weight_cache
    if _model_weight_cache is None:
        with _cache_init_lock:
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
    with _cache_init_lock:
        previous = _model_weight_cache
        _model_weight_cache = ModelWeightCache(max_cached_models=max_cached_models)
        if previous is not None:
            with previous._lock:
                _model_weight_cache._cache = previous._cache.copy()
            while len(_model_weight_cache._cache) > max_cached_models:
                evicted = _model_weight_cache._cache.popitem(last=False)[1]
                release_model(evicted)


def reset_model_weight_cache() -> None:
    """Clear and reset the process-wide cache (for tests)."""
    global _model_weight_cache
    with _cache_init_lock:
        if _model_weight_cache is not None:
            _model_weight_cache.clear()
        _model_weight_cache = None
