from __future__ import annotations

import importlib
from typing import Any

from timecopilot_gift_eval.protocol import ForecasterProtocol

from .jobs import load_models_config


def _import_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_model(model_key: str) -> ForecasterProtocol:
    models = load_models_config()
    if model_key not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model_key {model_key!r}. Available: {available}")

    spec = models[model_key]
    model_cls = _import_class(spec["class"])
    kwargs: dict[str, Any] = dict(spec.get("kwargs", {}))
    reference = spec.get("reference_slug")
    if reference is not None and "alias" not in kwargs:
        kwargs["alias"] = reference
    return model_cls(**kwargs)


def predictor_batch_size(model_key: str, *, default: int = 1024) -> int:
    spec = load_models_config()[model_key]
    if "predictor_batch_size" in spec:
        return int(spec["predictor_batch_size"])
    return default


def predictor_max_length(
    model_key: str,
    forecaster: ForecasterProtocol,
    *,
    default: int = 4096,
) -> int | None:
    spec = load_models_config()[model_key]
    if "max_length" in spec:
        max_length = spec["max_length"]
        return None if max_length is None else int(max_length)
    return int(getattr(forecaster, "context_length", default))


def reference_slug(model_key: str) -> str | None:
    models = load_models_config()
    if model_key not in models:
        raise KeyError(f"Unknown model_key {model_key!r}")
    return models[model_key].get("reference_slug")
