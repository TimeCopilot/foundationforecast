from __future__ import annotations

import importlib
from typing import Any

from .jobs import load_models_config
from foundationforecast.core.forecaster import Forecaster


def _import_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_model(model_key: str) -> Forecaster:
    models = load_models_config()
    if model_key not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown model_key {model_key!r}. Available: {available}")

    spec = models[model_key]
    model_cls = _import_class(spec["class"])
    kwargs: dict[str, Any] = dict(spec.get("kwargs", {}))
    if "alias" not in kwargs:
        kwargs["alias"] = model_key
    return model_cls(**kwargs)


def reference_csv(model_key: str) -> str | None:
    models = load_models_config()
    if model_key not in models:
        raise KeyError(f"Unknown model_key {model_key!r}")
    return models[model_key].get("reference_csv")
