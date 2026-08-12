import sys

from .chronos import Chronos, ChronosFinetuningConfig
from .moirai import Moirai
from .timegpt import TimeGPT, TimeGPTFinetuningConfig
from .timesfm import TimesFM
from .toto import Toto

__all__ = [
    "Chronos",
    "ChronosFinetuningConfig",
    "Moirai",
    "TimeGPT",
    "TimeGPTFinetuningConfig",
    "TimesFM",
    "Toto",
]

if sys.version_info >= (3, 11):
    __all__.append("TiRex")

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    __all__.extend(["Sundial", "TabPFN"])
