import sys

from foundationforecast.core.conformal import ConformalIntervals
from foundationforecast.models.chronos import Chronos
from foundationforecast.models.moirai import Moirai
from foundationforecast.models.tabby import Tabby
from foundationforecast.models.tafsut import Tafsut
from foundationforecast.models.timesfm import TimesFM
from foundationforecast.models.timestral import Timestral
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
    TimesFM(
        repo_id="google/timesfm-3.0-pytorch",
        context_length=256,
    ),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-2.0-R-small",
    ),
    Tafsut(context_length=512, batch_size=2),
    Tabby(context_length=512, batch_size=2),
    Timestral(
        repo_id="mistralai/Timestral-5M",
        alias="Timestral-5M",
        context_length=256,
        batch_size=2,
    ),
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
    models.append(
        T0(
            repo_id="theforecastingcompany/t0-beta",
            alias="t0-beta",
            context_length=256,
            batch_size=2,
        )
    )
    models.append(
        T0(
            repo_id="theforecastingcompany/t0-gamma",
            alias="t0-gamma",
            context_length=256,
            batch_size=2,
            trim_leading_missing=True,
            symmetric_averaging=True,
            clip_nonnegative=True,
            snap_integers=True,
            cross_inference=2,
        )
    )

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
    models.append(
        PatchTSTFM(
            repo_id="ibm-granite/granite-timeseries-patchtst-fm-r2",
            alias="Granite-PatchTST-FM-r2",
            context_length=2_048,
            batch_size=2,
        )
    )

if sys.version_info >= (3, 12):
    from foundationforecast.models.timeh2o import TimeH2O

    models.append(TimeH2O(context_length=256, batch_size=2))
    models.append(
        TimeH2O(
            repo_id="h2oai/timeh2o-v1-tiny",
            alias="TimeH2O-tiny",
            batch_size=2,
        )
    )

if sys.version_info < (3, 13):
    from contextlib import contextmanager

    import numpy as np
    import pandas as pd
    from tabpfn_time_series import TimeSeriesDataFrame

    from foundationforecast.models.tabpfn import TabPFN

    class _MockTabPFNPredictor:
        def predict(self, train_tsdf, test_tsdf, quantiles=None):
            result = {"target": np.full(len(test_tsdf), 1.0)}
            if quantiles is not None:
                for q in quantiles:
                    result[q] = np.full(len(test_tsdf), q)
            return TimeSeriesDataFrame(pd.DataFrame(result, index=test_tsdf.index))

    @contextmanager
    def _mock_get_model(_self):
        yield _MockTabPFNPredictor()

    tabpfn_model = TabPFN()
    tabpfn_model._get_model = lambda: _mock_get_model(tabpfn_model)  # type: ignore[method-assign, assignment]
    models.append(tabpfn_model)


def _with_conformal(model):
    model.prediction_intervals = ConformalIntervals(n_windows=2)
    return model


conformal_models = [
    _with_conformal(Tafsut(context_length=512, batch_size=2, alias="Tafsut-CF")),
    _with_conformal(
        Chronos(
            repo_id="amazon/chronos-bolt-tiny",
            alias="Chronos-Bolt-CF",
            batch_size=2,
        )
    ),
]
