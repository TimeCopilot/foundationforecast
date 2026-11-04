"""Drop-in replacement for `prophet.Prophet` backed by any foundation model.

Requires the optional `prophet` dependency:

```bash
pip install "foundationforecast[prophet]"
```
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        f"{e}. To use `foundationforecast.prophet` install the optional "
        'dependency: `pip install "foundationforecast[prophet]"`.'
    ) from e

from .core.forecaster import Forecaster

__all__ = ["FoundationProphet"]

logger = logging.getLogger(__name__)

# Placeholder id used to feed the single Prophet series to the panel API of
# the wrapped model. It never appears in any output.
_UNIQUE_ID = "uid_0"
_DEFAULT_INSAMPLE_WINDOWS = 10
_MIN_INSAMPLE_CONTEXT = 10


def _bind_model(cls: type, model: Forecaster) -> type:
    """Return a subclass of `cls` that remembers `model`.

    `prophet.diagnostics.cross_validation` clones a fitted model with
    `m.__class__(**prophet_kwargs)`, which knows nothing about `model`. Binding
    the model to the instance class lets that clone reuse it.
    """
    return type(
        cls.__name__,
        (cls,),
        {
            "_bound_model": model,
            "_unbound_class": cls,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__doc__": cls.__doc__,
        },
    )


def _restore(cls: type, model: Forecaster) -> Any:
    """Unpickle helper that rebuilds an instance of the model-bound class."""
    obj: Any = object.__new__(cls)
    obj.__class__ = _bind_model(cls, model)
    return obj


class FoundationProphet(Prophet):
    """`prophet.Prophet` with a foundation model instead of the Stan model.

    Use it to swap a foundation model into an existing Prophet pipeline: the
    `fit`, `make_future_dataframe`, `predict` and `plot` calls keep working and
    `predict` returns the usual `ds`, `yhat`, `yhat_lower` and `yhat_upper`
    columns.

    ```python
    import numpy as np
    import pandas as pd

    from foundationforecast.models import Chronos
    from foundationforecast.prophet import FoundationProphet

    ds = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"ds": ds, "y": 10 + np.sin(np.arange(120) * 2 * np.pi / 7)})

    model = Chronos(repo_id="amazon/chronos-bolt-tiny")
    m = FoundationProphet(model=model)  # instead of Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=14, include_history=False)
    forecast = m.predict(future)
    assert list(forecast.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ```

    Foundation models are zero-shot, so `fit` only stores the history. The
    forecast is produced in `predict`, and the interval covers
    `interval_width` of the predictive distribution.

    Historical dates in the dataframe given to `predict` (what
    `make_future_dataframe(include_history=True)` returns, and Prophet's
    default) have no fitted values. With `insample="cv"` they are filled with
    rolling-origin backtest forecasts from `model.cross_validation`, which
    runs the model once per window. A message is logged the first time this
    happens. Use `insample="none"` or `make_future_dataframe(include_history=False)`
    to skip it; historical rows are then `NaN`.

    Columns registered with `add_regressor` are passed to the model as
    known-future exogenous variables, so the wrapped model must support them
    (see the exogenous variables guide). Prophet-specific options (`growth`,
    `holidays`, seasonalities, changepoints, ...) are accepted for
    compatibility but have no effect. `plot_components` and the posterior
    sampling methods are not available.

    The history must be on a regular grid: its frequency is inferred from `ds`
    and the dates requested in `predict` must lie on that grid.
    """

    history: Any
    fit_kwargs: dict[str, Any]
    _bound_model: Forecaster | None = None
    _unbound_class: type | None = None

    def __init__(
        self,
        model: Forecaster | None = None,
        interval_width: float = 0.80,
        insample: Literal["cv", "none"] = "cv",
        insample_h: int | None = None,
        insample_n_windows: int | None = None,
        **prophet_kwargs: Any,
    ) -> None:
        """Create the adapter.

        Args:
            model: Foundation model (any `Forecaster`) that produces the
                forecasts, e.g. `Chronos()`.
            interval_width: Width of the prediction interval reported in
                `yhat_lower` and `yhat_upper`, as in Prophet. Converted to the
                model's `level` (`0.8` becomes `80`).
            insample: How to fill historical dates in `predict`. `"cv"` uses
                rolling-origin backtest forecasts; `"none"` leaves them `NaN`.
            insample_h: Horizon of each backtest window. Defaults to the
                number of future periods in `predict`, or to the seasonality of
                the frequency when there are none.
            insample_n_windows: Maximum number of backtest windows, counted
                back from the end of the history. Defaults to 10. Fewer
                windows are used when the history is short.
            **prophet_kwargs: Remaining `prophet.Prophet` arguments. They are
                forwarded to Prophet but do not affect the foundation model.
        """
        if model is None:
            model = type(self)._bound_model
        if model is None:
            raise TypeError("FoundationProphet requires a `model`, e.g. Chronos().")
        if not isinstance(model, Forecaster):
            raise TypeError(
                f"`model` must be a foundationforecast model, got {type(model)!r}."
            )
        if insample not in ("cv", "none"):
            raise ValueError(f"`insample` must be 'cv' or 'none', got {insample!r}.")
        for name, value in (
            ("insample_h", insample_h),
            ("insample_n_windows", insample_n_windows),
        ):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"`{name}` must be a positive integer.")
        level = round(100 * interval_width, 6)
        if not 0 < level < 100 or not float(level).is_integer():
            raise ValueError(
                "`interval_width` must be between 0 and 1 and a whole percent "
                f"(e.g. 0.8 or 0.95), got {interval_width!r}."
            )
        super().__init__(interval_width=interval_width, **prophet_kwargs)
        self._level = int(level)
        self.model = model
        self.insample = insample
        self.insample_h = insample_h
        self.insample_n_windows = insample_n_windows
        self._freq: str | None = None
        self._insample_logged = False
        if type(self)._bound_model is None:
            self.__class__ = _bind_model(type(self), model)

    def __reduce__(self) -> tuple[Any, ...]:
        base = type(self)._unbound_class or type(self)
        return (_restore, (base, self.model), self.__dict__.copy())

    @property
    def _regressors(self) -> list[str]:
        return list(self.extra_regressors)

    def add_regressor(self, name: str, *args: Any, **kwargs: Any) -> FoundationProphet:
        """Register a known-future exogenous variable passed to the model."""
        if kwargs.get("regressor_predictor"):
            raise NotImplementedError(
                "`regressor_predictor` is not supported: future regressor values "
                "must be provided in the dataframe given to `predict`."
            )
        return super().add_regressor(name, *args, **kwargs)

    def fit(self, df: pd.DataFrame, **kwargs: Any) -> FoundationProphet:
        """Store the history; foundation models need no training.

        Args:
            df: Dataframe with the dates in `ds`, the values in `y`, and one
                column per regressor added with `add_regressor`.
            **kwargs: Accepted for compatibility with `Prophet.fit` and ignored.

        Returns:
            The fitted adapter.
        """
        if self.history is not None:
            raise Exception(
                "Prophet object can only be fit once. Instantiate a new object."
            )
        if "ds" not in df or "y" not in df:
            raise ValueError(
                'Dataframe must have columns "ds" and "y" with the dates and '
                "values respectively."
            )
        history = df[df["y"].notnull()].copy()
        if history.shape[0] < 2:
            raise ValueError("Dataframe has less than 2 non-NaN rows.")
        history["ds"] = self._check_ds(history["ds"])
        if history["ds"].duplicated().any():
            raise ValueError("Column 'ds' has duplicated dates.")
        for name in self._regressors:
            self._check_regressor(history, name)
        history = history.sort_values("ds").reset_index(drop=True)
        self._freq = self._infer_freq(history["ds"])
        self.history_dates = pd.to_datetime(
            pd.Series(df["ds"].unique(), name="ds")
        ).sort_values()
        self.history = history[["ds", "y", *self._regressors]]
        self.fit_kwargs = {}
        return self

    def predict(
        self, df: pd.DataFrame | None = None, vectorized: bool = True
    ) -> pd.DataFrame:
        """Forecast the dates in `df`.

        Args:
            df: Dataframe with the dates in `ds` and one column per regressor.
                Dates after the history must continue it on its frequency, as
                `make_future_dataframe` does. Defaults to the historical dates.
            vectorized: Accepted for compatibility with `Prophet.predict` and
                ignored.

        Returns:
            Dataframe with `ds`, `yhat`, `yhat_lower` and `yhat_upper`, one row
                per row of `df`. Historical dates without a backtest forecast
                are `NaN`.
        """
        if self.history is None or self._freq is None:
            raise Exception("Model has not been fit.")
        history = self.history
        if df is None:
            future = history[["ds", *self._regressors]].copy()
        else:
            if df.shape[0] == 0:
                raise ValueError("Dataframe has no rows.")
            if "ds" not in df:
                raise ValueError('Dataframe must have a column "ds" with the dates.')
            future = df.copy()
            future["ds"] = self._check_ds(future["ds"])
        last_date = history["ds"].max()
        is_future = future["ds"] > last_date

        parts: list[pd.DataFrame] = []
        grid = pd.DatetimeIndex([])
        h = 0
        if is_future.any():
            grid = pd.date_range(
                start=last_date, end=future["ds"].max(), freq=self._freq
            )[1:]
            h = len(grid)
            off_grid = ~future.loc[is_future, "ds"].isin(grid)
            if off_grid.any():
                bad = future.loc[is_future, "ds"][off_grid].iloc[0]
                raise ValueError(
                    f"Date {bad} is not on the frequency '{self._freq}' of the "
                    "history. Dates after the history must continue it, e.g. "
                    "use `make_future_dataframe`."
                )
            parts.append(self._forecast_future(future, grid))
        if self.insample == "cv" and (~is_future).any():
            insample = self._forecast_insample(h)
            if insample is not None:
                parts.append(insample)

        columns = ["yhat", "yhat_lower", "yhat_upper"]
        if parts:
            fcst = pd.concat(parts, ignore_index=True).drop_duplicates("ds")
        else:
            fcst = pd.DataFrame(
                {"ds": pd.DatetimeIndex([]), **dict.fromkeys(columns, [])}
            )
        out = future[["ds"]].merge(fcst, on="ds", how="left")
        return out[["ds", *columns]]

    def plot_components(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FoundationProphet has no trend or seasonality components to plot."
        )

    def predict_uncertainty(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FoundationProphet does not sample trajectories; intervals come from "
            "the foundation model through `interval_width`."
        )

    def sample_posterior_predictive(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FoundationProphet does not sample posterior predictive trajectories."
        )

    def predictive_samples(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "FoundationProphet does not sample posterior predictive trajectories."
        )

    @staticmethod
    def _check_ds(ds: pd.Series) -> pd.Series:
        ds = pd.to_datetime(ds)
        if ds.isnull().any():
            raise ValueError("Found NaN in column 'ds'.")
        if ds.dt.tz is not None:
            raise ValueError(
                "Column 'ds' has timezone specified, which is not supported. "
                "Remove the timezone."
            )
        return ds

    @staticmethod
    def _check_regressor(df: pd.DataFrame, name: str) -> None:
        if name not in df:
            raise ValueError(f"Regressor {name!r} missing from dataframe.")
        if df[name].isnull().any():
            raise ValueError(f"Found NaN in column {name!r}.")

    @staticmethod
    def _infer_freq(ds: pd.Series) -> str:
        freq = None
        for candidate in (ds, ds.tail(5)):
            try:
                freq = pd.infer_freq(candidate)
            except ValueError:
                continue
            if freq is not None:
                return freq
        raise ValueError(
            "Unable to infer the frequency of `ds`. The history must be on a "
            "regular grid with at least 3 dates."
        )

    def _panel_history(self) -> pd.DataFrame:
        assert self.history is not None
        panel = self.history.copy()
        panel.insert(0, "unique_id", _UNIQUE_ID)
        return panel

    def _to_prophet(self, fcst: pd.DataFrame) -> pd.DataFrame:
        alias = self.model.alias
        lo = [c for c in fcst.columns if c.startswith(f"{alias}-lo-")]
        hi = [c for c in fcst.columns if c.startswith(f"{alias}-hi-")]
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(fcst["ds"]).astype("datetime64[ns]").to_numpy(),
                "yhat": fcst[alias].to_numpy(dtype=float),
                "yhat_lower": fcst[lo[0]].to_numpy(dtype=float) if lo else np.nan,
                "yhat_upper": fcst[hi[0]].to_numpy(dtype=float) if hi else np.nan,
            }
        )

    def _forecast_future(
        self, future: pd.DataFrame, grid: pd.DatetimeIndex
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        regressors = self._regressors
        if regressors:
            for name in regressors:
                if name not in future:
                    raise ValueError(f"Regressor {name!r} missing from dataframe.")
            x_df = pd.DataFrame({"ds": grid}).merge(
                future[["ds", *regressors]].drop_duplicates("ds"),
                on="ds",
                how="left",
            )
            for name in regressors:
                if x_df[name].isnull().any():
                    raise ValueError(
                        f"Regressor {name!r} needs a value for every period from "
                        f"{grid[0]} to {grid[-1]}."
                    )
            x_df.insert(0, "unique_id", _UNIQUE_ID)
            kwargs = {"X_df": x_df, "futr_exog_list": regressors}
        fcst = self.model.forecast(
            df=self._panel_history(),
            h=len(grid),
            freq=self._freq,
            level=[self._level],
            **kwargs,
        )
        return self._to_prophet(fcst)

    def _forecast_insample(self, h_future: int) -> pd.DataFrame | None:
        assert self.history is not None and self._freq is not None
        h = self.insample_h or h_future
        if not h:
            h = max(1, int(self.model._maybe_get_seasonality(self._freq)))
        max_windows = (len(self.history) - _MIN_INSAMPLE_CONTEXT) // h
        n_windows = min(
            self.insample_n_windows or _DEFAULT_INSAMPLE_WINDOWS, max_windows
        )
        if n_windows < 1:
            logger.warning(
                "FoundationProphet: the history is too short to compute in-sample "
                "forecasts (horizon %d); historical rows are returned as NaN.",
                h,
            )
            return None
        if not self._insample_logged:
            self._insample_logged = True
            logger.warning(
                "FoundationProphet: computing in-sample forecasts with %d "
                "rolling-origin windows of horizon %d using %s, which runs the "
                "model %d extra time(s). Pass insample='none' to the constructor "
                "or use make_future_dataframe(include_history=False) to skip "
                "them, or set insample_n_windows / insample_h to change them.",
                n_windows,
                h,
                self.model.alias,
                n_windows,
            )
        regressors = self._regressors
        cv = self.model.cross_validation(
            df=self._panel_history(),
            h=h,
            freq=self._freq,
            n_windows=n_windows,
            step_size=h,
            level=[self._level],
            futr_exog_list=regressors or None,
        )
        return self._to_prophet(cv)
