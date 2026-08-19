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
    kwargs: dict[str, Any] = spec.get("kwargs", {})
    return model_cls(**kwargs)


def reference_slug(model_key: str) -> str | None:
    models = load_models_config()
    if model_key not in models:
        raise KeyError(f"Unknown model_key {model_key!r}")
    return models[model_key].get("reference_slug")
