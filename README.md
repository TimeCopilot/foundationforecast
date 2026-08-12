# foundationforecast

Foundation time series forecasting models, isolated from TimeCopilot.

```python
from foundationforecast import FoundationForecast
from foundationforecast.models import Chronos, Toto

ff = FoundationForecast(models=[Chronos(), Toto()])
fcst = ff.forecast(df, h=12, freq="D")
```
