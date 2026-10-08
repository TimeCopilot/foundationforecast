from __future__ import annotations

import importlib
from typing import Any, Literal

from timecopilot_gift_eval.protocol import ForecasterProtocol

from .jobs import load_models_config_for_registry

Registry = Literal["default", "replication"]


def _import_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _model_spec(model_key: str, registry: Registry) -> dict:
    models = load_models_config_for_registry(registry)
    if model_key not in models:
        available = ", ".join(sorted(models))
        raise KeyError(
            f"Unknown model_key {model_key!r} in registry {registry!r}. "
            f"Available: {available}"
        )
    return models[model_key]


def model_spec(model_key: str, *, registry: Registry = "default") -> dict:
    return _model_spec(model_key, registry)


def build_model(
    model_key: str,
    *,
    registry: Registry = "default",
) -> ForecasterProtocol:
    spec = _model_spec(model_key, registry)
    model_cls = _import_class(spec["class"])
    kwargs: dict[str, Any] = dict(spec.get("kwargs", {}))
    reference = spec.get("reference_slug")
    if reference is not None and "alias" not in kwargs:
        kwargs["alias"] = reference
    return model_cls(**kwargs)


def predictor_batch_size(
    model_key: str,
    *,
    registry: Registry = "default",
    default: int = 1024,
) -> int:
    spec = _model_spec(model_key, registry)
    if "predictor_batch_size" in spec:
        return int(spec["predictor_batch_size"])
    return default


def predictor_max_length(
    model_key: str,
    forecaster: ForecasterProtocol,
    *,
    registry: Registry = "default",
    default: int = 4096,
) -> int | None:
    spec = _model_spec(model_key, registry)
    if "max_length" in spec:
        max_length = spec["max_length"]
        return None if max_length is None else int(max_length)
    return int(getattr(forecaster, "context_length", default))


def reference_slug(model_key: str, *, registry: Registry = "default") -> str | None:
    return _model_spec(model_key, registry).get("reference_slug")


def model_keys_with_reference(*, registry: Registry = "default") -> list[str]:
    models = load_models_config_for_registry(registry)
    return [
        model_key
        for model_key, spec in models.items()
        if spec.get("reference_slug") is not None
    ]
