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
    from .tirex import TiRex

    __all__.append("TiRex")

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    from .flowstate import FlowState
    from .patchtst_fm import PatchTSTFM
    from .t0 import T0

    __all__.extend(["FlowState", "PatchTSTFM", "T0"])

if sys.version_info < (3, 13):
    from .sundial import Sundial
    from .tabpfn import TabPFN

    __all__.extend(["Sundial", "TabPFN"])
