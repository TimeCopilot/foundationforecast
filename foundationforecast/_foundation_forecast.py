from __future__ import annotations

from .core.forecaster import Forecaster
from .core.multi_model import MultiModelForecasterMixin
from .core.panel_columns import (
    CANONICAL_ID_COL,
    CANONICAL_TARGET_COL,
    CANONICAL_TIME_COL,
)


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
        *,
        id_col: str = CANONICAL_ID_COL,
        time_col: str = CANONICAL_TIME_COL,
        target_col: str = CANONICAL_TARGET_COL,
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
        super().__init__(
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        if not models:
            raise ValueError("At least one model is required.")
        self._validate_unique_aliases(models)
        self.models = models
        self.fallback_model = fallback_model
        self.clean_cache = clean_cache
