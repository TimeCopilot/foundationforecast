import sys

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

if sys.version_info >= (3, 11):
    from .models import TiRex as TiRex

    __all__.append("TiRex")

if (3, 11) <= sys.version_info < (3, 14):
    from .models import T0 as T0
    from .models import FlowState as FlowState
    from .models import PatchTSTFM as PatchTSTFM

    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    from .models import Sundial as Sundial
    from .models import TabPFN as TabPFN

    __all__.extend(["Sundial", "TabPFN"])
