import inspect

import numpy as np
import pandas as pd
import pytest
from utilsforecast.processing import make_future_dataframe

import foundationforecast.models as ff_models
from foundationforecast import ACI, PID, FoundationForecast
from foundationforecast.core.conformal import (
    ConformalIntervals,
    add_conformal_distribution_intervals,
)
from foundationforecast.core.forecaster import ExogCapableForecaster, Forecaster
from foundationforecast.core.utils import process_panel_from_df


class _MeanForecaster(ExogCapableForecaster):
    """Deterministic forecaster: point forecast = training mean per series."""

    def __init__(
        self,
        *,
        alias: str = "Mean",
        prediction_intervals: ConformalIntervals | None = None,
    ) -> None:
        super().__init__(prediction_intervals=prediction_intervals)
        self.alias = alias

    def _forecast_univariate(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        level,
        quantiles,
        panel,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        if panel is None:
            panel = process_panel_from_df(df)
        futr = make_future_dataframe(
            uids=panel.uids,
            last_times=pd.to_datetime(panel.last_times),
            freq=freq,
            h=h,
            id_col="unique_id",
            time_col="ds",
        )
        futr[self.alias] = np.repeat([arr.mean() for arr in panel.series_arrays], h)
        return futr


def _panel(n_series: int = 2, length: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        for t in range(length):
            rows.append(
                {
                    "unique_id": str(i),
                    "ds": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                    "y": float(i + 1 + (t % 5)),
                }
            )
    return pd.DataFrame(rows)


def test_conformal_calibration_keeps_prediction_intervals():
    class _AssertingForecaster(_MeanForecaster):
        def cross_validation(self, *args, **kwargs):
            assert self.prediction_intervals is not None
            return super().cross_validation(*args, **kwargs)

    intervals = ConformalIntervals(n_windows=2)
    model = _AssertingForecaster(prediction_intervals=intervals)
    fcst = model.forecast(_panel(), h=3, freq="D", level=[80])
    assert model.prediction_intervals is intervals
    assert "Mean-lo-80" in fcst.columns


def test_conformal_without_level_returns_point_forecast():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    fcst = model.forecast(_panel(), h=3, freq="D")
    assert list(fcst.columns) == ["unique_id", "ds", "Mean"]


def test_conformal_median_only_quantiles_skip_calibration():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    fcst = model.forecast(_panel(length=10), h=3, freq="D", quantiles=[0.5])
    assert "Mean-q-50" in fcst.columns
    assert (fcst["Mean-q-50"] == fcst["Mean"]).all()
    assert not any("-lo-" in c or "-hi-" in c for c in fcst.columns)


def test_detect_anomalies_with_prediction_intervals():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    result = model.detect_anomalies(_panel(length=40), h=3, freq="D")
    assert "Mean-anomaly" in result.columns


def test_conformal_ignores_history_only_exog_columns():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(length=40)
    df["noise"] = 1.0
    fcst = model.forecast(df=df, h=3, freq="D", level=[80])
    assert "Mean-lo-80" in fcst.columns


@pytest.mark.parametrize(
    "method",
    ["conformal_distribution", "conformal_error_finite_sample", PID(ncal=2)],
)
def test_conformal_staggered_cutoffs_match_per_series(method):
    rows = []
    for uid, start, length in (("a", 0, 30), ("b", 20, 30)):
        for t in range(length):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": pd.Timestamp("2020-01-01") + pd.Timedelta(days=start + t),
                    "y": float(t % 5),
                }
            )
    df = pd.DataFrame(rows)
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=4, method=method),
    )
    joint = model.forecast(df=df, h=3, freq="D", level=[80])
    per_series = pd.concat(
        [
            model.forecast(df=sdf, h=3, freq="D", level=[80])
            for _, sdf in df.groupby("unique_id")
        ],
        ignore_index=True,
    )
    cols = ["unique_id", "ds", "Mean", "Mean-lo-80", "Mean-hi-80"]
    pd.testing.assert_frame_equal(
        joint[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        per_series[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_conformal_forecast_adds_interval_columns():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(length=40)
    fcst = model.forecast(df=df, h=3, freq="D", level=[80, 90])
    assert "Mean-lo-80" in fcst.columns
    assert "Mean-hi-90" in fcst.columns
    assert not any("-q-" in c for c in fcst.columns)


@pytest.mark.parametrize(
    "method",
    ["conformal_distribution", "conformal_error_finite_sample", PID(ncal=2)],
)
def test_conformal_forecast_through_foundation_forecast_matches_direct(method):
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=4, method=method),
    )
    df = _panel(n_series=3, length=40)
    direct = model.forecast(df=df, h=3, freq="D", level=[80])
    via_ff = FoundationForecast(models=[model]).forecast(
        df=df, h=3, freq="D", level=[80]
    )
    pd.testing.assert_frame_equal(
        direct.reset_index(drop=True),
        via_ff[direct.columns].reset_index(drop=True),
    )


def test_conformal_forecast_matches_point_only_cross_validation():
    df = _panel(n_series=2, length=40)
    h, n_windows = 3, 2
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=n_windows),
    )
    fcst = model.forecast(df=df, h=h, freq="D", level=[80])
    cv = _MeanForecaster().cross_validation(
        df=df, h=h, freq="D", n_windows=n_windows, step_size=h
    )
    cs_df = cv[["unique_id", "ds", "cutoff", "Mean"]].copy()
    cs_df["Mean"] = (cv["Mean"] - cv["y"]).abs()
    cs_df = cs_df.sort_values(["unique_id", "cutoff", "ds"])
    cs_df["window"] = cs_df.groupby("unique_id", sort=False)["cutoff"].rank(
        method="dense"
    )
    cs_df = cs_df.sort_values(["window", "unique_id", "ds"])
    expected = add_conformal_distribution_intervals(
        fcst[["unique_id", "ds", "Mean"]].sort_values(["unique_id", "ds"]),
        cs_df,
        model_names=["Mean"],
        level=[80],
        cs_n_windows=n_windows,
        cs_h=h,
        n_series=2,
        horizon=h,
    )
    pd.testing.assert_frame_equal(
        fcst.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "method",
    ["conformal_distribution", "conformal_error_finite_sample", PID(ncal=2)],
)
def test_cross_validation_recalibrates_conformal_intervals_per_window(method):
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=4, method=method),
    )
    df = _panel(n_series=2, length=60)
    cv = model.cross_validation(df=df, h=3, freq="D", n_windows=2, level=[80])
    for cutoff, window in cv.groupby("cutoff"):
        expected = model.forecast(df[df["ds"] <= cutoff], h=3, freq="D", level=[80])
        pd.testing.assert_frame_equal(
            window[expected.columns].reset_index(drop=True),
            expected.reset_index(drop=True),
        )
    assert cv.groupby("cutoff")["Mean-hi-80"].mean().nunique() == 2


def test_finite_sample_quantiles_match_level_bounds_and_keep_median():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(
            n_windows=9, method="conformal_error_finite_sample"
        ),
    )
    df = _panel(length=40)
    levels = model.forecast(df, h=2, freq="D", level=[80, 90, 95])
    quantiles = model.forecast(
        df, h=2, freq="D", quantiles=[0.025, 0.1, 0.5, 0.9, 0.975]
    )
    for q, bound in [
        ("2.5", "lo-95"),
        ("10", "lo-80"),
        ("90", "hi-80"),
        ("97.5", "hi-95"),
    ]:
        np.testing.assert_array_equal(quantiles[f"Mean-q-{q}"], levels[f"Mean-{bound}"])
    np.testing.assert_array_equal(quantiles["Mean-q-50"], levels["Mean"])
    assert np.isfinite(levels[["Mean-lo-90", "Mean-hi-90"]]).all().all()
    assert np.isneginf(quantiles["Mean-q-2.5"]).all()
    assert np.isposinf(quantiles["Mean-q-97.5"]).all()
    assert not any("-lo-" in col or "-hi-" in col for col in quantiles)


@pytest.mark.parametrize(
    "kwargs", [{"level": [80, 95]}, {"quantiles": [0.025, 0.1, 0.9, 0.975]}]
)
def test_finite_sample_forecast_warns_for_infinite_bounds(kwargs):
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(
            n_windows=2, method="conformal_error_finite_sample"
        )
    )
    with pytest.warns(UserWarning) as caught:
        result = model.forecast(_panel(), h=2, freq="D", **kwargs)
    user_warnings = [warning for warning in caught if warning.category is UserWarning]
    assert len(user_warnings) == 1
    message = str(user_warnings[0].message)
    assert "level 80 needs at least 4 calibration windows, got 2" in message
    assert "level 95 needs at least 19 calibration windows, got 2" in message
    assert np.isfinite(result["Mean"]).all()
    assert np.isinf(result.drop(columns=["unique_id", "ds", "Mean"])).all().all()


def test_conformal_forecast_with_integer_unique_ids():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    )
    df = _panel(length=40)
    df["unique_id"] = df["unique_id"].astype(int)
    fcst = model.forecast(df=df, h=3, freq="D", level=[80])
    assert fcst["Mean-lo-80"].notna().all()
    assert fcst["Mean-hi-80"].notna().all()


def test_conformal_error_method_adds_columns():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2, method="conformal_error"),
    )
    fcst = model.forecast(_panel(length=40), h=3, freq="D", level=[80])
    assert "Mean-lo-80" in fcst.columns
    assert (fcst["Mean-hi-80"] >= fcst["Mean"]).all()


def test_conformal_step_size_defaults_to_horizon():
    df = _panel(length=40)
    default = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2),
    ).forecast(df=df, h=3, freq="D", level=[80])
    explicit = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=2, step_size=3),
    ).forecast(df=df, h=3, freq="D", level=[80])
    pd.testing.assert_frame_equal(
        default.reset_index(drop=True), explicit.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        getattr(ff_models, name)
        for name in ff_models.__all__
        if isinstance(getattr(ff_models, name), type)
        and issubclass(getattr(ff_models, name), Forecaster)
    ],
)
def test_model_constructors_accept_prediction_intervals(model_cls):
    intervals = ConformalIntervals(n_windows=3)
    model = model_cls(prediction_intervals=intervals)
    assert model.prediction_intervals is intervals


@pytest.mark.parametrize(
    "model_cls",
    [
        cls
        for name in ("Simons", "T0", "Tabby", "TimeH2O", "Timestral")
        if (cls := getattr(ff_models, name, None)) is not None
    ],
)
def test_prediction_intervals_appended_after_model_options(model_cls):
    params = list(inspect.signature(model_cls.__init__).parameters)
    assert params[-1] == "prediction_intervals"


def test_native_level_without_prediction_intervals():
    model = _MeanForecaster()
    df = _panel(length=10)
    fcst = model.forecast(df=df, h=2, freq="D", level=[80])
    assert "Mean-lo-80" not in fcst.columns


@pytest.mark.parametrize("step_size", [None, 1])
@pytest.mark.parametrize("symmetric", [False, True])
def test_pid_forecast_matches_replay_of_signed_cross_validation_errors(
    step_size, symmetric
):
    df = _panel(n_series=2, length=60)
    h, n_windows = 3, 6
    level: list[int | float] = [80, 95]
    method = PID(ncal=3, symmetric=symmetric)
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(
            n_windows=n_windows, method=method, step_size=step_size
        ),
    )
    fcst = model.forecast(df=df, h=h, freq="D", level=level)
    cv = _MeanForecaster().cross_validation(
        df=df, h=h, freq="D", n_windows=n_windows, step_size=step_size or h
    )
    cv = cv.sort_values(["unique_id", "cutoff", "ds"])
    errors = (cv["y"] - cv["Mean"]).to_numpy().reshape(2, n_windows, h)
    lower, upper = method.radii(
        errors.transpose(1, 0, 2), level, step_size or h, final_gap=h
    )
    fcst = fcst.sort_values(["unique_id", "ds"])
    mean = fcst["Mean"].to_numpy().reshape(2, h)
    for i, lv in enumerate(level):
        np.testing.assert_allclose(
            fcst[f"Mean-lo-{lv}"].to_numpy().reshape(2, h), mean - lower[i]
        )
        np.testing.assert_allclose(
            fcst[f"Mean-hi-{lv}"].to_numpy().reshape(2, h), mean + upper[i]
        )


def test_pid_adapts_to_trailing_level_shift():
    # Same model and horizon: PID tracks the error scale of recent windows, so
    # a volatility jump in the latest windows widens the interval relative to
    # a calm history.
    rng = np.random.default_rng(0)

    def _series(scale_after: float) -> pd.DataFrame:
        noise = rng.normal(size=80)
        noise[60:] *= scale_after
        return pd.DataFrame(
            {
                "unique_id": "a",
                "ds": pd.date_range("2020-01-01", periods=80, freq="D"),
                "y": noise,
            }
        )

    widths = {}
    for scale in (1.0, 5.0):
        model = _MeanForecaster(
            prediction_intervals=ConformalIntervals(
                n_windows=10, method=PID(ncal=4), step_size=2
            ),
        )
        fcst = model.forecast(_series(scale), h=2, freq="D", level=[80])
        widths[scale] = (fcst["Mean-hi-80"] - fcst["Mean-lo-80"]).mean()
    assert widths[5.0] > widths[1.0]


@pytest.mark.parametrize("step_size", [None, 1])
def test_aci_forecast_matches_causal_replay_and_foundation_forecast(step_size):
    df = _panel(n_series=2, length=60)
    method = ACI(gamma=0.005, ncal=4)
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(
            n_windows=12, method=method, step_size=step_size
        )
    )
    forecast = model.forecast(df, h=3, freq="D", level=[80, 95])
    cv = (
        _MeanForecaster()
        .cross_validation(df, h=3, freq="D", n_windows=12, step_size=step_size or 3)
        .sort_values(["unique_id", "cutoff", "ds"])
    )
    errors = (cv["y"] - cv["Mean"]).to_numpy().reshape(2, 12, 3).transpose(1, 0, 2)
    radii = np.maximum.accumulate(
        method.thresholds(errors, [80, 95], step_size or 3, 3), axis=0
    )
    mean = forecast["Mean"].to_numpy().reshape(2, 3)
    for i, level in enumerate([80, 95]):
        np.testing.assert_allclose(
            forecast[f"Mean-lo-{level}"], (mean - radii[i]).ravel()
        )
        np.testing.assert_allclose(
            forecast[f"Mean-hi-{level}"], (mean + radii[i]).ravel()
        )
    via_ff = FoundationForecast(models=[model]).forecast(
        df, h=3, freq="D", level=[80, 95]
    )
    pd.testing.assert_frame_equal(forecast, via_ff[forecast.columns])


def test_aci_cross_validation_restarts_at_each_cutoff():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=6, method=ACI(gamma=0.005))
    )
    df = _panel(length=50)
    result = model.cross_validation(df, h=2, freq="D", n_windows=3, level=[80, 95])
    for cutoff, window in result.groupby("cutoff"):
        expected = model.forecast(df[df["ds"] <= cutoff], h=2, freq="D", level=[80, 95])
        pd.testing.assert_frame_equal(
            window[expected.columns].reset_index(drop=True), expected
        )


def test_aci_quantiles_use_the_matching_bounds_and_point_median():
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=6, method=ACI(gamma=0.005))
    )
    df = _panel()
    levels = model.forecast(df, h=2, freq="D", level=[80, 95])
    quantiles = model.forecast(
        df, h=2, freq="D", quantiles=[0.025, 0.1, 0.5, 0.9, 0.975]
    )
    for q, suffix in [
        ("2.5", "lo-95"),
        ("10", "lo-80"),
        ("90", "hi-80"),
        ("97.5", "hi-95"),
    ]:
        np.testing.assert_array_equal(
            quantiles[f"Mean-q-{q}"], levels[f"Mean-{suffix}"]
        )
    np.testing.assert_array_equal(quantiles["Mean-q-50"], levels["Mean"])


@pytest.mark.parametrize("kwargs", [{}, {"quantiles": [0.5]}])
@pytest.mark.filterwarnings("error:ACI replay.*:UserWarning")
def test_aci_point_only_forecasts_do_not_calibrate_or_warn(kwargs, monkeypatch):
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(n_windows=20, method=ACI())
    )
    # Too little history for calibration is fine when no interval is requested.
    monkeypatch.setattr(
        model, "cross_validation", lambda **kw: pytest.fail("unneeded calibration")
    )
    result = model.forecast(_panel(length=5), h=2, freq="D", **kwargs)
    assert np.isfinite(result["Mean"]).all()


@pytest.mark.parametrize("empty", [False, True])
def test_aci_forecast_warns_about_nonfinite_sets(empty):
    df = _panel(n_series=1, length=40)
    df["y"] = 1.0 if empty else np.arange(len(df), dtype=float)
    model = _MeanForecaster(
        prediction_intervals=ConformalIntervals(
            n_windows=4 if empty else 12,
            method=ACI(gamma=1 if empty else 0.5, ncal=1 if empty else 2),
        )
    )
    kind, level = ("empty", 50) if empty else ("unbounded", 95)
    with pytest.warns(UserWarning, match=f"ACI.*{kind}"):
        result = model.forecast(df, h=1, freq="D", level=[level])
    if empty:
        assert np.isposinf(result[f"Mean-lo-{level}"]).all()
        assert np.isneginf(result[f"Mean-hi-{level}"]).all()
    else:
        assert np.isneginf(result[f"Mean-lo-{level}"]).all()
        assert np.isposinf(result[f"Mean-hi-{level}"]).all()
