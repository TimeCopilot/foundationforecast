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
    from .models import TiRex

    __all__.append("TiRex")

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    from .models import FlowState, PatchTSTFM, T0

    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    from .models import Sundial, TabPFN

    __all__.extend(["Sundial", "TabPFN"])
