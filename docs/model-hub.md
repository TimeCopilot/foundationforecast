# Foundation Model Hub

**foundationforecast** provides a unified API for state-of-the-art foundation models for time series forecasting. Run multiple pretrained models through [`FoundationForecast`](api/forecaster.md) with minimal code changes.

!!! note "Family example notebooks"
    Example notebooks for foundation model families are in [Examples](examples/index.md#foundation-models).

!!! tip "Forecast multiple models using a unified API"

    ```python
    import pandas as pd
    from foundationforecast import FoundationForecast
    from foundationforecast.models import Chronos, Toto

    df = pd.read_csv(
        "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
        parse_dates=["ds"],
    )
    ff = FoundationForecast(
        models=[
            Chronos(),
            Toto(context_length=256),
        ]
    )

    fcst_df = ff.forecast(df=df, h=12, freq="MS")
    cv_df = ff.cross_validation(df=df, h=12, freq="MS")
    ```

---

## Foundation Models

Below is the list of available foundation models. Click a model name for API details.

- [Chronos](api/models/foundation/models.md#foundationforecast.models.chronos.Chronos) ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815))
- [FlowState](api/models/foundation/models.md#foundationforecast.models.flowstate.FlowState) ([arXiv:2508.05287](https://arxiv.org/abs/2508.05287)) — Python 3.11–3.13
- [Moirai](api/models/foundation/models.md#foundationforecast.models.moirai.Moirai) ([arXiv:2402.02592](https://arxiv.org/abs/2402.02592))
- [PatchTST-FM](api/models/foundation/models.md#foundationforecast.models.patchtst_fm.PatchTSTFM) ([arXiv:2602.06909](https://arxiv.org/abs/2602.06909)) — Python 3.11–3.13
- [Sundial](api/models/foundation/models.md#foundationforecast.models.sundial.Sundial) ([arXiv:2502.00816](https://arxiv.org/pdf/2502.00816))
- [T0](api/models/foundation/models.md#foundationforecast.models.t0.T0) ([model card](https://huggingface.co/theforecastingcompany/t0-alpha)) — Python 3.11–3.13
- [TabPFN](api/models/foundation/models.md#foundationforecast.models.tabpfn.TabPFN) ([arXiv:2501.02945](https://arxiv.org/abs/2501.02945)) — Python 3.10–3.12
- [Tafsut](api/models/foundation/models.md#foundationforecast.models.tafsut.Tafsut) ([GitHub](https://github.com/Tafsut-FM/tafsut))
- [TiRex / TiRex-2](api/models/foundation/models.md#foundationforecast.models.tirex.TiRex) ([arXiv:2505.23719](https://arxiv.org/abs/2505.23719), [arXiv:2607.01204](https://arxiv.org/abs/2607.01204)) — Python 3.11+
- [TimeGPT](api/models/foundation/models.md#foundationforecast.models.timegpt.TimeGPT) ([arXiv:2310.03589](https://arxiv.org/abs/2310.03589)) — requires `NIXTLA_API_KEY`
- [TimesFM](api/models/foundation/models.md#foundationforecast.models.timesfm.TimesFM) ([arXiv:2310.10688](https://arxiv.org/abs/2310.10688))
- [Toto](api/models/foundation/models.md#foundationforecast.models.toto.Toto) ([arXiv:2505.14766](https://arxiv.org/abs/2505.14766))
