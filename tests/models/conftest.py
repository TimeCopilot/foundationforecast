import sys

from foundationforecast.models.chronos import Chronos
from foundationforecast.models.moirai import Moirai
from foundationforecast.models.tafsut import Tafsut
from foundationforecast.models.timesfm import TimesFM
from foundationforecast.models.toto import Toto

models = [
    Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
    Chronos(repo_id="amazon/chronos-2", alias="Chronos-2"),
    Chronos(repo_id="amazon/chronos-2", alias="Chronos-2", batch_size=2),
    Toto(context_length=256, batch_size=2),
    Toto(
        repo_id="Datadog/Toto-2.0-4m",
        alias="Toto-2",
        context_length=256,
        batch_size=2,
    ),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-1.1-R-small",
    ),
    TimesFM(
        repo_id="google/timesfm-1.0-200m-pytorch",
        context_length=256,
    ),
    TimesFM(
        repo_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
    ),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-2.0-R-small",
    ),
    Tafsut(context_length=512, batch_size=2),
]

if sys.version_info >= (3, 11):
    from foundationforecast.models.tirex import TiRex

    models.append(TiRex())
    models.append(
        TiRex(
            repo_id="NX-AI/TiRex-2",
            alias="TiRex-2",
            batch_size=2,
        )
    )

if (3, 11) <= sys.version_info < (3, 14):
    from foundationforecast.models.t0 import T0

    models.append(T0(context_length=256, batch_size=2))

if (3, 11) <= sys.version_info < (3, 14):
    from foundationforecast.models.flowstate import FlowState
    from foundationforecast.models.patchtst_fm import PatchTSTFM

    models.append(FlowState(repo_id="ibm-research/flowstate"))
    models.append(
        FlowState(
            repo_id="ibm-granite/granite-timeseries-flowstate-r1",
            alias="FlowState-Granite",
        )
    )
    models.append(PatchTSTFM(context_length=2_048))

if sys.version_info < (3, 13):
    from tabpfn_time_series import TabPFNMode

    from foundationforecast.models.tabpfn import TabPFN

    models.append(TabPFN(mode=TabPFNMode.MOCK))
