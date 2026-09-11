from __future__ import annotations

from .core.forecaster import Forecaster
from .core.multi_model import MultiModelForecasterMixin


class FoundationForecast(MultiModelForecasterMixin, Forecaster):
    """Unified forecaster for multiple foundation time series models.

    This class runs multiple pretrained foundation
    models through a single interface and merges their forecasts.
    """

    alias = "FoundationForecast"

    def __init__(
        self,
        models: list[Forecaster],
        fallback_model: Forecaster | None = None,
        clean_cache: bool = False,
    ):
        """Run multiple foundation models through one interface.

        Args:
            models: Forecasters to run; each must have a unique ``alias``.
            fallback_model: Optional substitute when a model raises
                ``ValueError`` or ``RuntimeError``.
            clean_cache: When ``True``, call ``clear_model_cache()`` on every
                model after each one finishes forecasting. Helps limit peak GPU
                memory in multi-model ensembles.
        """
        if not models:
            raise ValueError("At least one model is required.")
        self._validate_unique_aliases(models)
        self.models = models
        self.fallback_model = fallback_model
        self.clean_cache = clean_cache
