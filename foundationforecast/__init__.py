from ._foundation_forecast import FoundationForecast
from .models import (
    Chronos,
    ChronosFinetuningConfig,
    Moirai,
    TimeGPT,
    TimeGPTFinetuningConfig,
    TimesFM,
    Toto,
)

__all__ = [
    "FoundationForecast",
    "Chronos",
    "ChronosFinetuningConfig",
    "Moirai",
    "TimeGPT",
    "TimeGPTFinetuningConfig",
    "TimesFM",
    "Toto",
]

import sys

if sys.version_info >= (3, 11):
    __all__.append("TiRex")

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    __all__.extend(["Sundial", "TabPFN"])
