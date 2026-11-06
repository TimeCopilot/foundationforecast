import copy
import logging
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet")

from foundationforecast.core.conformal import ConformalIntervals  # noqa: E402
from foundationforecast.core.forecaster import Forecaster  # noqa: E402
from foundationforecast.prophet import FoundationProphet  # noqa: E402


class NaiveForecaster(Forecaster):
    """Last value (+ first regressor) with deterministic `level` bounds.

    Not a foundation model: it only honours the `Forecaster` contract, which is
    all the adapter relies on.
    """

    alias = "Naive"

    def __init__(self) -> None:
        super().__init__()
        self.forecast_calls: list[dict] = []

    def forecast(
        self,
        df,
        h,
        freq=None,
        level=None,
        quantiles=None,
        panel=None,
        X_df=None,
        *,
        futr_df=None,
        futr_exog_list=None,
    ):
        self.validate_input(df, h)
        self.forecast_calls.append(
            {
                "h": h,
                "freq": freq,
                "level": level,
                "X_df": X_df,
                "futr_exog_list": futr_exog_list,
                "unique_ids": set(df["unique_id"]),
            }
        )
        freq = self._maybe_infer_freq(df, freq)
        last = df.sort_values("ds").iloc[-1]
        ds = pd.date_range(last["ds"], periods=h + 1, freq=freq)[1:]
        yhat = np.full(h, float(last["y"]))
        if X_df is not None:
            yhat = yhat + X_df.sort_values("ds")[futr_exog_list[0]].to_numpy()
        out = pd.DataFrame(
            {"unique_id": df["unique_id"].iloc[0], "ds": ds, self.alias: yhat}
        )
        for lv in level or []:
            out[f"{self.alias}-lo-{lv}"] = yhat - lv / 10
            out[f"{self.alias}-hi-{lv}"] = yhat + lv / 10
        return out


def make_df(n=60, freq="D", with_x=False):
    ds = pd.date_range("2024-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"ds": ds, "y": np.arange(n, dtype=float)})
    if with_x:
        df["x"] = np.arange(n, dtype=float) % 5
    return df


def fitted(df=None, **kwargs):
    model = NaiveForecaster()
    m = FoundationProphet(model=model, **kwargs)
    m.fit(make_df() if df is None else df)
    return m, model


def test_fit_predict_future_only():
    m, model = fitted()
    future = m.make_future_dataframe(periods=7, include_history=False)
    fcst = m.predict(future)

    assert list(fcst.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    pd.testing.assert_series_equal(fcst["ds"], future["ds"])
    np.testing.assert_allclose(fcst["yhat"], 59.0)
    np.testing.assert_allclose(fcst["yhat_lower"], 59.0 - 8.0)
    np.testing.assert_allclose(fcst["yhat_upper"], 59.0 + 8.0)
    # one forecast call, no backtest
    assert len(model.forecast_calls) == 1
    call = model.forecast_calls[0]
    assert call["h"] == 7
    assert call["freq"] == "D"
    assert call["unique_ids"] == {"uid_0"}


@pytest.mark.parametrize("interval_width, level", [(0.8, 80), (0.95, 95), (0.9, 90)])
def test_interval_width_maps_to_level(interval_width, level):
    m, model = fitted(interval_width=interval_width)
    m.predict(m.make_future_dataframe(periods=3, include_history=False))
    assert model.forecast_calls[0]["level"] == [level]


@pytest.mark.parametrize("interval_width", [0.975, 0, 1, 1.5])
def test_invalid_interval_width(interval_width):
    with pytest.raises(ValueError, match="interval_width"):
        FoundationProphet(model=NaiveForecaster(), interval_width=interval_width)


def test_predict_defaults_to_history_dates():
    m, _ = fitted()
    fcst = m.predict()
    assert len(fcst) == 60
    assert fcst["yhat"].notna().sum() > 0


@pytest.mark.parametrize("freq", ["MS", "h", "W-MON"])
def test_other_frequencies(freq):
    df = make_df(n=48, freq=freq)
    m, model = fitted(df)
    future = m.make_future_dataframe(periods=5, freq=freq, include_history=False)
    fcst = m.predict(future)
    # pandas aliases differ across versions (e.g. "H" vs "h"), so compare to inference
    assert model.forecast_calls[0]["freq"] == pd.infer_freq(df["ds"])
    assert fcst["yhat"].notna().all()
    pd.testing.assert_series_equal(fcst["ds"], future["ds"])


def test_future_subset_of_horizon_is_aligned():
    m, _ = fitted(make_df(n=30))
    future = m.make_future_dataframe(periods=5, include_history=False)
    subset = future.iloc[[3, 1]]
    fcst = m.predict(subset)
    assert fcst["ds"].tolist() == subset["ds"].tolist()
    assert fcst["yhat"].notna().all()


def test_off_grid_future_dates_raise():
    m, _ = fitted()
    future = pd.DataFrame({"ds": [pd.Timestamp("2024-03-15 12:00")]})
    with pytest.raises(ValueError, match="not on the frequency"):
        m.predict(future)


def test_insample_cv_fills_history(caplog):
    m, model = fitted(make_df(n=60))
    future = m.make_future_dataframe(periods=7)  # include_history=True
    with caplog.at_level(logging.WARNING, logger="foundationforecast.prophet"):
        fcst = m.predict(future)

    assert len(fcst) == 67
    # h=7 backtest windows over history: (60 - 10) // 7 = 7 windows -> last 49 rows
    assert fcst["yhat"].isna().sum() == 11
    assert fcst["yhat"].iloc[:11].isna().all()
    assert fcst["yhat"].iloc[11:].notna().all()
    # naive backtest forecast = last value before each cutoff
    hist = fcst.iloc[:60].reset_index(drop=True)
    np.testing.assert_allclose(hist["yhat"].iloc[11:18], 10.0)
    np.testing.assert_allclose(hist["yhat"].iloc[18:25], 17.0)
    covered = fcst.dropna()
    assert (covered["yhat_lower"] <= covered["yhat"]).all()
    assert (covered["yhat"] <= covered["yhat_upper"]).all()
    assert len(model.forecast_calls) == 1 + 7

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "insample='none'" in messages[0]
    assert "include_history=False" in messages[0]
    assert "7 rolling-origin windows" in messages[0]

    # logged once per instance
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="foundationforecast.prophet"):
        m.predict(future)
    assert caplog.records == []


def test_insample_none_leaves_history_nan(caplog):
    m, model = fitted(insample="none")
    with caplog.at_level(logging.WARNING, logger="foundationforecast.prophet"):
        fcst = m.predict(m.make_future_dataframe(periods=7))
    assert fcst["yhat"].iloc[:60].isna().all()
    assert fcst["yhat"].iloc[60:].notna().all()
    assert len(model.forecast_calls) == 1
    assert caplog.records == []


def test_insample_window_options():
    m, model = fitted(insample_h=5, insample_n_windows=3)
    fcst = m.predict(m.make_future_dataframe(periods=7))
    assert fcst["yhat"].iloc[:60].notna().sum() == 15
    cv_calls = model.forecast_calls[1:]
    assert len(cv_calls) == 3
    assert {c["h"] for c in cv_calls} == {5}


def test_insample_h_defaults_to_seasonality_without_future():
    m, model = fitted(make_df(n=60, freq="MS"))
    fcst = m.predict()
    # MS seasonality is 12: (60 - 10) // 12 = 4 windows of 12 steps
    assert {c["h"] for c in model.forecast_calls} == {12}
    assert fcst["yhat"].notna().sum() == 48


def test_insample_short_history_warns_and_returns_nan(caplog):
    m, model = fitted(make_df(n=12))
    with caplog.at_level(logging.WARNING, logger="foundationforecast.prophet"):
        fcst = m.predict(m.make_future_dataframe(periods=5))
    assert "too short" in caplog.records[0].getMessage()
    assert fcst["yhat"].iloc[:12].isna().all()
    assert fcst["yhat"].iloc[12:].notna().all()
    assert len(model.forecast_calls) == 1


def test_regressors_flow_to_model_as_known_future_exog():
    model = NaiveForecaster()
    m = FoundationProphet(model=model, insample="none")
    m.add_regressor("x")
    df = make_df(with_x=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=4, include_history=False)
    future["x"] = [10.0, 20.0, 30.0, 40.0]
    fcst = m.predict(future)

    call = model.forecast_calls[0]
    assert call["futr_exog_list"] == ["x"]
    assert list(call["X_df"].columns) == ["unique_id", "ds", "x"]
    np.testing.assert_allclose(fcst["yhat"], 59.0 + np.array([10, 20, 30, 40]))


def test_regressors_are_used_in_insample_backtest():
    model = NaiveForecaster()
    m = FoundationProphet(model=model, insample_n_windows=2)
    m.add_regressor("x")
    m.fit(make_df(with_x=True))
    future = m.make_future_dataframe(periods=4)
    future["x"] = 0.0
    m.predict(future)
    cv_calls = model.forecast_calls[1:]
    assert len(cv_calls) == 2
    assert all(c["futr_exog_list"] == ["x"] for c in cv_calls)


def test_regressor_errors():
    m = FoundationProphet(model=NaiveForecaster(), insample="none")
    m.add_regressor("x")
    with pytest.raises(ValueError, match="Regressor 'x' missing"):
        m.fit(make_df())
    m = FoundationProphet(model=NaiveForecaster(), insample="none")
    m.add_regressor("x")
    m.fit(make_df(with_x=True))
    future = m.make_future_dataframe(periods=3, include_history=False)
    with pytest.raises(ValueError, match="Regressor 'x' missing"):
        m.predict(future)
    future["x"] = [1.0, np.nan, 3.0]
    with pytest.raises(ValueError, match="needs a value for every period"):
        m.predict(future)


def test_regressor_predictor_not_supported():
    m = FoundationProphet(model=NaiveForecaster())
    with pytest.raises(NotImplementedError, match="regressor_predictor"):
        m.add_regressor("x", regressor_predictor=True)


def test_fit_validation():
    model = NaiveForecaster()
    with pytest.raises(ValueError, match='columns "ds" and "y"'):
        FoundationProphet(model=model).fit(pd.DataFrame({"ds": [1]}))
    with pytest.raises(ValueError, match="less than 2 non-NaN rows"):
        FoundationProphet(model=model).fit(make_df(n=1))
    with pytest.raises(ValueError, match="duplicated"):
        FoundationProphet(model=model).fit(pd.concat([make_df(5), make_df(5)]))
    tz = make_df(10)
    tz["ds"] = tz["ds"].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone"):
        FoundationProphet(model=model).fit(tz)
    irregular = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-05", "2024-02-20", "2024-03-30"]
            ),
            "y": np.arange(5, dtype=float),
        }
    )
    with pytest.raises(ValueError, match="regular grid"):
        FoundationProphet(model=model).fit(irregular)


def test_fit_drops_trailing_null_y_and_fits_once():
    df = make_df(30)
    df.loc[29, "y"] = np.nan
    model = NaiveForecaster()
    m = FoundationProphet(model=model, insample="none")
    m.fit(df)
    assert len(m.history) == 29
    assert len(m.history_dates) == 30
    # the dropped last period is part of the horizon, as in Prophet
    future = m.make_future_dataframe(periods=3, include_history=False)
    assert future["ds"].iloc[0] == pd.Timestamp("2024-01-31")
    fcst = m.predict(future)
    assert model.forecast_calls[0]["h"] == 4
    assert fcst["yhat"].notna().all()
    with pytest.raises(Exception, match="only be fit once"):
        m.fit(df)


def test_fit_rejects_gaps_in_history():
    gap = make_df(30).drop(index=[10, 11])
    with pytest.raises(ValueError, match="regular grid"):
        FoundationProphet(model=NaiveForecaster()).fit(gap)
    # an interior null `y` is dropped, which leaves a gap
    null = make_df(30)
    null.loc[5, "y"] = np.nan
    with pytest.raises(ValueError, match="regular grid"):
        FoundationProphet(model=NaiveForecaster()).fit(null)


def test_history_keeps_extra_columns_but_model_does_not_see_them():
    model = NaiveForecaster()
    df = make_df().assign(cap=100.0, extra="a")
    m = FoundationProphet(model=model, insample="none")
    m.fit(df)
    assert {"cap", "extra"} <= set(m.history)
    m.predict(m.make_future_dataframe(periods=3, include_history=False))
    assert list(m._panel_history().columns) == ["unique_id", "ds", "y"]


def test_diagnostics_with_logistic_growth():
    from prophet.diagnostics import cross_validation

    m, _ = fitted(make_df(n=90).assign(cap=1000.0), growth="logistic", insample="none")
    df_cv = cross_validation(
        m, horizon="7 days", period="14 days", initial="45 days", disable_tqdm=True
    )
    assert df_cv["yhat"].notna().all()


def test_interval_width_can_change_after_construction():
    m, model = fitted()
    m.interval_width = 0.95
    m.predict(m.make_future_dataframe(periods=3, include_history=False))
    assert model.forecast_calls[-1]["level"] == [95]
    m.interval_width = 0.7
    m.predict(m.make_future_dataframe(periods=3, include_history=False))
    assert model.forecast_calls[-1]["level"] == [70]
    m.interval_width = 0.975
    with pytest.raises(ValueError, match="interval_width"):
        m.predict(m.make_future_dataframe(periods=3, include_history=False))


def test_interval_width_applies_to_insample_backtest():
    m, model = fitted(insample_n_windows=2)
    m.interval_width = 0.9
    m.predict(m.make_future_dataframe(periods=7))
    assert {tuple(c["level"]) for c in model.forecast_calls} == {(90,)}


def test_insample_windows_reserve_conformal_calibration_samples(caplog):
    model = NaiveForecaster()
    model.prediction_intervals = ConformalIntervals(n_windows=2)
    m = FoundationProphet(model=model)
    m.fit(make_df(n=120))
    m.predict(m.make_future_dataframe(periods=14))
    # conformal needs 14 + 14 + 1 = 29 samples: (120 - 29) // 14 = 6 windows
    assert len(model.forecast_calls) == 1 + 6
    first_cutoff_train = 120 - 6 * 14
    assert first_cutoff_train >= 29

    short = NaiveForecaster()
    short.prediction_intervals = ConformalIntervals(n_windows=2)
    m = FoundationProphet(model=short)
    m.fit(make_df(n=40))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="foundationforecast.prophet"):
        fcst = m.predict(m.make_future_dataframe(periods=14))
    assert "too short" in caplog.records[0].getMessage()
    assert fcst["yhat"].iloc[:40].isna().all()
    assert fcst["yhat"].iloc[40:].notna().all()


def test_predict_before_fit():
    with pytest.raises(Exception, match="has not been fit"):
        FoundationProphet(model=NaiveForecaster()).predict(make_df(3))


def test_constructor_validation():
    with pytest.raises(TypeError, match="requires a `model`"):
        FoundationProphet()
    with pytest.raises(TypeError, match="must be a foundationforecast model"):
        FoundationProphet(model="chronos")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="insample"):
        FoundationProphet(
            model=NaiveForecaster(),
            insample="always",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="insample_h"):
        FoundationProphet(model=NaiveForecaster(), insample_h=0)


def test_prophet_arguments_are_accepted_with_a_warning():
    with pytest.warns(UserWarning, match="ignores") as record:
        m = FoundationProphet(
            model=NaiveForecaster(),
            growth="linear",
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
            n_changepoints=5,
        )
    assert m.seasonality_mode == "multiplicative"
    assert m.n_changepoints == 5
    message = str(record[0].message)
    # only the arguments that differ from Prophet's defaults are listed
    assert "`yearly_seasonality`" in message
    assert "`seasonality_mode`" in message
    assert "`n_changepoints`" in message
    assert "`growth`" not in message


def test_ignored_holidays_dataframe_warns():
    holidays = pd.DataFrame({"holiday": ["x"], "ds": pd.to_datetime(["2024-01-05"])})
    with pytest.warns(UserWarning, match="`holidays`"):
        FoundationProphet(model=NaiveForecaster(), holidays=holidays)


def test_no_warning_for_default_and_supported_arguments():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FoundationProphet(
            model=NaiveForecaster(),
            interval_width=0.9,
            insample="none",
            growth="linear",
            seasonality_mode="additive",
        )


def test_add_seasonality_and_country_holidays_warn():
    m = FoundationProphet(model=NaiveForecaster())
    with pytest.warns(UserWarning, match="add_seasonality"):
        m.add_seasonality("monthly", period=30.5, fourier_order=3)
    with pytest.warns(UserWarning, match="add_country_holidays"):
        m.add_country_holidays("US")


def test_unsupported_methods_raise():
    m, _ = fitted()
    fcst = m.predict(m.make_future_dataframe(periods=3, include_history=False))
    with pytest.raises(NotImplementedError, match="components"):
        m.plot_components(fcst)
    with pytest.raises(NotImplementedError):
        m.predict_uncertainty(fcst)
    with pytest.raises(NotImplementedError):
        m.sample_posterior_predictive(fcst)
    with pytest.raises(NotImplementedError):
        m.predictive_samples(fcst)


def test_plot():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    m, _ = fitted()
    fcst = m.predict(m.make_future_dataframe(periods=7))
    fig = m.plot(fcst)
    assert fig is not None


def test_prophet_diagnostics_cross_validation():
    from prophet.diagnostics import cross_validation, performance_metrics

    m, model = fitted(make_df(n=90), insample="none")
    df_cv = cross_validation(
        m, horizon="7 days", period="14 days", initial="45 days", disable_tqdm=True
    )
    assert {"ds", "yhat", "yhat_lower", "yhat_upper", "y", "cutoff"} <= set(df_cv)
    assert df_cv["yhat"].notna().all()
    # the clones reuse the wrapped model
    assert len(model.forecast_calls) == df_cv["cutoff"].nunique()
    metrics = performance_metrics(df_cv)
    assert "mae" in metrics


def test_deepcopy_and_pickle_keep_the_model():
    m, _ = fitted()
    for clone in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert isinstance(clone, FoundationProphet)
        assert isinstance(clone.model, NaiveForecaster)
        future = clone.make_future_dataframe(periods=3, include_history=False)
        assert clone.predict(future)["yhat"].notna().all()
        # prophet_copy-style cloning (no `model` argument) reuses the model
        assert type(clone)(interval_width=0.9).model is clone.model


def test_rebinding_a_bound_class_to_another_model():
    m, model = fitted()
    other_model = NaiveForecaster()
    other = type(m)(model=other_model)
    assert other.model is other_model
    # clones of `other` use `other_model`, not the model `m` was bound to
    assert type(other)(interval_width=0.9).model is other_model
    assert type(m)(interval_width=0.9).model is model
    assert type(other).__name__ == "FoundationProphet"


def test_integer_dates_raise():
    df = make_df(30).assign(ds=np.arange(30))
    with pytest.raises(ValueError, match="must contain dates"):
        FoundationProphet(model=NaiveForecaster()).fit(df)
    m, _ = fitted()
    with pytest.raises(ValueError, match="must contain dates"):
        m.predict(pd.DataFrame({"ds": [60, 61]}))


def test_numeric_string_targets_are_converted():
    from prophet.diagnostics import cross_validation, performance_metrics

    df = make_df(n=90).assign(y=lambda d: d["y"].astype(str))
    m, _ = fitted(df, insample="none")
    assert pd.api.types.is_float_dtype(m.history["y"])
    df_cv = cross_validation(
        m, horizon="7 days", period="14 days", initial="45 days", disable_tqdm=True
    )
    assert "mae" in performance_metrics(df_cv)


def test_bound_class_keeps_public_name():
    m, _ = fitted()
    assert type(m).__name__ == "FoundationProphet"
    assert isinstance(m, FoundationProphet)


def test_missing_prophet_gives_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "foundationforecast.prophet")
    monkeypatch.setitem(sys.modules, "prophet", None)
    with pytest.raises(ModuleNotFoundError, match=r"foundationforecast\[prophet\]"):
        __import__("foundationforecast.prophet")


@pytest.mark.models
def test_chronos_bolt_end_to_end():
    from foundationforecast.models import Chronos

    ds = pd.date_range("2024-01-01", periods=120, freq="D")
    y = 10 + np.sin(np.arange(120) * 2 * np.pi / 7)
    m = FoundationProphet(
        model=Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
        interval_width=0.9,
        insample_n_windows=2,
    )
    m.fit(pd.DataFrame({"ds": ds, "y": y}))
    future = m.make_future_dataframe(periods=14)
    fcst = m.predict(future)

    assert len(fcst) == 134
    # history: 120 rows; the last 2 windows of 14 steps are backtested
    assert fcst["yhat"].iloc[:92].isna().all()
    assert fcst["yhat"].iloc[92:].notna().all()
    tail = fcst.iloc[-14:]
    assert (tail["yhat_lower"] <= tail["yhat"]).all()
    assert (tail["yhat"] <= tail["yhat_upper"]).all()


@pytest.mark.models
def test_chronos_2_regressors_end_to_end():
    from foundationforecast.models import Chronos

    n = 96
    ds = pd.date_range("2024-01-01", periods=n + 7, freq="D")
    x = np.sin(np.arange(n + 7) * 2 * np.pi / 7)
    y = 5 + 3 * x
    m = FoundationProphet(
        model=Chronos(repo_id="amazon/chronos-2", alias="Chronos-2"),
        insample="none",
    )
    m.add_regressor("x")
    m.fit(pd.DataFrame({"ds": ds[:n], "y": y[:n], "x": x[:n]}))
    future = pd.DataFrame({"ds": ds[n:], "x": x[n:]})
    fcst = m.predict(future)
    assert fcst["yhat"].notna().all()
    assert np.corrcoef(fcst["yhat"], y[n:])[0, 1] > 0.5
