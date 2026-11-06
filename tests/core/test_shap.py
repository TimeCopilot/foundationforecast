import itertools
import math

import numpy as np
import pandas as pd
import pytest

from foundationforecast.core.forecaster import Forecaster
from foundationforecast.core.shap import (
    ShapExplanations,
    auto_history,
    estimate_series_forecasts,
    resolve_method,
)

H = 6
N_ROWS = 14 * 24
START = pd.Timestamp("2024-01-01")


class FakeForecaster(Forecaster):
    """Deterministic model: recent level, a long-run term, exog and an interaction."""

    alias = "Fake"

    def __init__(self, nan_ok: bool = True, exog: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.supports_nan_target = nan_ok
        self._exog = exog
        self.calls: list[pd.DataFrame] = []

    def supports_native_futr_exog(self) -> bool:
        return self._exog

    def forecast(  # type: ignore[override]
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
        self.calls.append(df.copy())
        horizon = X_df if X_df is not None else futr_df
        cols = futr_exog_list or []
        rows = []
        for uid, g in df.sort_values(["unique_id", "ds"]).groupby(
            "unique_id", sort=False
        ):
            y = g["y"].to_numpy(dtype=float)
            if not self.supports_nan_target:
                assert not np.isnan(y).any(), "model received NaN targets"
            recent = np.nanmean(y[-6:]) if not np.isnan(y[-6:]).all() else 0.0
            old = np.nanmean(y) if not np.isnan(y).all() else 0.0
            last = pd.Timestamp(g["ds"].max())
            ds = pd.date_range(last, periods=h + 1, freq=freq or "h")[1:]
            vals = 2.0 * recent + 0.5 * old
            if horizon is not None:
                hz = horizon[horizon["unique_id"] == uid].sort_values("ds")
                for c in cols:
                    vals = vals + {"temp": 3.0, "hol": -2.0}[c] * hz[c].to_numpy()
                if "temp" in cols and "hol" in cols:
                    vals = vals + 0.5 * hz["temp"].to_numpy() * hz["hol"].to_numpy()
            rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, self.alias: vals}))
        return pd.concat(rows, ignore_index=True)


def make_data(n_series: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range(START, periods=N_ROWS, freq="h")
    hist, fut = [], []
    for i in range(n_series):
        y = 10 + 5 * i + np.sin(np.arange(N_ROWS) / 4) + rng.normal(size=N_ROWS) * 0.1
        hist.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": ds,
                    "y": y,
                    "temp": rng.normal(size=N_ROWS),
                    "hol": rng.integers(0, 2, N_ROWS).astype(float),
                }
            )
        )
        fds = pd.date_range(ds[-1] + pd.Timedelta(hours=1), periods=H, freq="h")
        fut.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": fds,
                    "temp": rng.normal(size=H),
                    "hol": rng.integers(0, 2, H).astype(float),
                }
            )
        )
    return pd.concat(hist, ignore_index=True), pd.concat(fut, ignore_index=True)


def shap_cols(out: pd.DataFrame, alias: str = "Fake") -> list[str]:
    return [c for c in out.columns if c.startswith(f"{alias}-shap-")]


def brute_force(value, n: int) -> np.ndarray:
    """Exact Shapley values from a coalition value function (independent oracle)."""
    phi = None
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for subset in itertools.combinations(others, r):
                s = np.zeros(n, dtype=bool)
                s[list(subset)] = True
                with_i = s.copy()
                with_i[i] = True
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                delta = w * (value(with_i) - value(s))
                if phi is None:
                    phi = np.zeros((n, len(delta)))
                phi[i] += delta
    assert phi is not None
    return phi


def oracle_value(model, df, fut, mask, baseline, cuts=(6, 24)):
    """Value of a coalition, masking the panel without using library internals."""
    n_hist = len(cuts) + 1
    hist_on, temp_on = mask[:n_hist], mask[n_hist]
    out = []
    for uid, g in df.groupby("unique_id", sort=False):
        pos = np.arange(len(g))[::-1]
        win = (pos[:, None] >= np.array(cuts)[None, :]).sum(axis=1)
        if not hist_on.any():
            out.append(np.full(H, baseline[uid]))
            continue
        keep = win <= np.flatnonzero(hist_on).max()
        y = g["y"].to_numpy().copy()
        y[keep & ~hist_on[win]] = np.nan
        part = g.loc[keep, ["unique_id", "ds"]].copy()
        part["y"] = y[keep]
        cols = ["temp", "hol"]
        part[cols] = g.loc[keep, cols].to_numpy()
        use = ["hol"] + (["temp"] if temp_on else [])
        hz = fut[fut["unique_id"] == uid]
        res = model.forecast(
            part, H, freq="h", X_df=hz[["unique_id", "ds", *use]], futr_exog_list=use
        )
        out.append(res["Fake"].to_numpy())
    return np.concatenate(out)


def test_config_validation():
    with pytest.raises(ValueError, match="method"):
        ShapExplanations(method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mask"):
        ShapExplanations(mask="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="offset"):
        ShapExplanations(history=["nope"])
    with pytest.raises(ValueError, match="positive"):
        ShapExplanations(history=[0])
    with pytest.raises(ValueError, match="at least one"):
        ShapExplanations(history=[])
    with pytest.raises(ValueError, match="exog"):
        ShapExplanations(exog="temp")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_samples"):
        ShapExplanations(n_samples=0)


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("h", ["1D", "7D"]),
        ("30min", ["1D", "7D"]),
        ("D", ["7D", "28D"]),
        ("W-SUN", [4, 13]),
        ("MS", [3, 12]),
        ("QE-DEC", [4, 12]),
        ("B", [5, 20]),
    ],
)
def test_auto_history_by_freq(freq, expected):
    assert auto_history(freq, 4) == expected


def test_resolve_method_follows_permutation_budget():
    assert resolve_method("auto", 7, 10) == "exact"
    assert resolve_method("auto", 8, 10) == "permutation"
    assert resolve_method("auto", 8, 100) == "exact"
    assert resolve_method("loo", 3, 10) == "loo"


def test_estimate_skips_coalitions_without_history():
    assert estimate_series_forecasts("exact", 3, 1, 2, 10) == (2**4 - 2) * 2
    assert estimate_series_forecasts("loo", 3, 1, 5, 10) == 5 * 5


def test_exact_matches_independent_brute_force_and_is_additive():
    df, fut = make_data()
    model = FakeForecaster()
    cfg = ShapExplanations(history=[6, 24], exog=["temp"], method="exact")
    out = model.explain(df, H, freq="h", X_df=fut, explanations=cfg)

    assert list(out.columns[:3]) == ["unique_id", "ds", "Fake"]
    assert shap_cols(out) == [
        "Fake-shap-base",
        "Fake-shap-hist-6",
        "Fake-shap-hist-6-24",
        "Fake-shap-hist-older",
        "Fake-shap-temp",
    ]
    assert len(out) == 2 * H
    total = out[shap_cols(out)].sum(axis=1)
    np.testing.assert_allclose(total, out["Fake"], atol=1e-8)

    baseline = df.groupby("unique_id")["y"].mean().to_dict()
    phi = brute_force(lambda m: oracle_value(model, df, fut, m, baseline), n=4)
    got = out[[c for c in shap_cols(out) if c != "Fake-shap-base"]].to_numpy().T
    np.testing.assert_allclose(got.reshape(4, -1), phi, atol=1e-8)

    full = model.forecast(
        df,
        H,
        freq="h",
        X_df=fut[["unique_id", "ds", "temp", "hol"]],
        futr_exog_list=["temp", "hol"],
    )
    np.testing.assert_allclose(out["Fake"], full["Fake"], atol=1e-8)


def test_non_player_exog_is_always_provided():
    df, fut = make_data(1)
    model = FakeForecaster()
    cfg = ShapExplanations(history=[6], exog=["temp"], method="exact")
    out = model.explain(df, H, freq="h", X_df=fut, explanations=cfg)
    full = model.forecast(df, H, freq="h", X_df=fut, futr_exog_list=["temp", "hol"])
    np.testing.assert_allclose(out["Fake"], full["Fake"], atol=1e-8)
    assert "Fake-shap-hol" not in out.columns
    assert all("hol" in c.columns for c in model.calls)


def test_exog_false_has_no_exog_players():
    df, fut = make_data(1)
    out = FakeForecaster().explain(
        df,
        H,
        freq="h",
        X_df=fut,
        explanations=ShapExplanations(history=[6], exog=False, method="exact"),
    )
    assert not any(c.endswith(("-temp", "-hol")) for c in out.columns)


def test_unknown_exog_player_raises():
    df, fut = make_data(1)
    with pytest.raises(ValueError, match="not among"):
        FakeForecaster().explain(
            df, H, freq="h", X_df=fut, explanations=ShapExplanations(exog=["nope"])
        )


def test_history_only_without_x_df():
    df, _ = make_data(2)
    out = FakeForecaster().explain(
        df[["unique_id", "ds", "y"]],
        H,
        freq="h",
        explanations=ShapExplanations(history=["6h"], method="exact"),
    )
    assert shap_cols(out) == [
        "Fake-shap-base",
        "Fake-shap-hist-6h",
        "Fake-shap-hist-older",
    ]
    total = out[shap_cols(out)].sum(axis=1)
    np.testing.assert_allclose(total, out["Fake"], atol=1e-8)


def test_permutation_is_additive_and_reproducible():
    pytest.importorskip("shap")
    df, fut = make_data()
    cfg = ShapExplanations(
        history=[6, 24], exog=True, method="permutation", n_samples=8, seed=0
    )
    a = FakeForecaster().explain(df, H, freq="h", X_df=fut, explanations=cfg)
    b = FakeForecaster().explain(df, H, freq="h", X_df=fut, explanations=cfg)
    pd.testing.assert_frame_equal(a, b)
    total = a[shap_cols(a)].sum(axis=1)
    np.testing.assert_allclose(total, a["Fake"], atol=1e-6)


def test_permutation_close_to_exact_on_small_game():
    df, fut = make_data(1)
    exact = ShapExplanations(history=[6, 24], exog=["temp"], method="exact")
    perm = ShapExplanations(
        history=[6, 24],
        exog=["temp"],
        method="permutation",
        n_samples=600,
        seed=1,
        warn_model_calls=10**9,
    )
    a = FakeForecaster().explain(df, H, freq="h", X_df=fut, explanations=exact)
    model = FakeForecaster()
    b = model.explain(df, H, freq="h", X_df=fut, explanations=perm)
    cols = shap_cols(a)
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), atol=0.15)
    # coalitions are memoised: no more series forecasts than distinct coalitions
    assert sum(c["unique_id"].nunique() for c in model.calls) <= 2**4


def test_loo_reports_residual():
    df, fut = make_data()
    model = FakeForecaster()
    out = model.explain(
        df,
        H,
        freq="h",
        X_df=fut,
        explanations=ShapExplanations(history=[6, 24], exog=True, method="loo"),
    )
    assert "Fake-shap-residual" in out.columns
    parts = [c for c in shap_cols(out)]
    np.testing.assert_allclose(out[parts].sum(axis=1), out["Fake"], atol=1e-8)
    assert out["Fake-shap-residual"].abs().max() > 0


def test_loo_does_not_need_shap(monkeypatch):
    import foundationforecast.core.shap as shap_mod

    def boom():
        raise AssertionError("shap must not be imported for loo")

    monkeypatch.setattr(shap_mod, "_import_shap", boom)
    df, _ = make_data(1)
    out = FakeForecaster().explain(
        df[["unique_id", "ds", "y"]],
        H,
        freq="h",
        explanations=ShapExplanations(history=[6], method="loo"),
    )
    assert "Fake-shap-residual" in out.columns


def test_missing_shap_raises_helpful_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no shap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df, _ = make_data(1)
    with pytest.raises(ImportError, match=r"foundationforecast\[explain\]"):
        FakeForecaster().explain(
            df[["unique_id", "ds", "y"]],
            H,
            freq="h",
            explanations=ShapExplanations(history=[6], method="exact"),
        )


def test_nan_masking_for_models_that_accept_nan():
    df, _ = make_data(1)
    model = FakeForecaster(nan_ok=True)
    model.explain(
        df[["unique_id", "ds", "y"]],
        H,
        freq="h",
        explanations=ShapExplanations(history=[6, 24], method="exact"),
    )
    assert any(c["y"].isna().any() for c in model.calls)


def test_fill_masking_uses_baseline_and_never_sends_nan():
    df, _ = make_data(1)
    model = FakeForecaster(nan_ok=False)
    model.explain(
        df[["unique_id", "ds", "y"]],
        H,
        freq="h",
        explanations=ShapExplanations(history=[6, 24], method="exact"),
    )
    baseline = df["y"].mean()
    filled = [c for c in model.calls if np.isclose(c["y"].to_numpy(), baseline).any()]
    assert filled
    assert all(not c["y"].isna().any() for c in model.calls)


def test_oldest_windows_are_truncated_from_the_front():
    df, _ = make_data(1)
    model = FakeForecaster()
    model.explain(
        df[["unique_id", "ds", "y"]],
        H,
        freq="h",
        explanations=ShapExplanations(
            history=[6, 24], method="exact", max_batch_series=1
        ),
    )
    lengths = {len(c) for c in model.calls}
    assert N_ROWS in lengths
    assert 6 in lengths
    assert 24 in lengths
    assert all(c["ds"].max() == df["ds"].max() for c in model.calls)


def test_ids_subset_and_unknown_ids():
    df, fut = make_data(3)
    out = FakeForecaster().explain(
        df,
        H,
        freq="h",
        X_df=fut,
        ids=["s1"],
        explanations=ShapExplanations(history=[6], method="exact"),
    )
    assert set(out["unique_id"]) == {"s1"}
    with pytest.raises(ValueError, match="ids not found"):
        FakeForecaster().explain(df, H, freq="h", X_df=fut, ids=["zzz"])


def test_batching_does_not_change_results():
    df, fut = make_data(3)
    big = ShapExplanations(history=[6, 24], method="exact")
    small = ShapExplanations(history=[6, 24], method="exact", max_batch_series=1)
    a = FakeForecaster().explain(df, H, freq="h", X_df=fut, explanations=big)
    model = FakeForecaster()
    b = model.explain(df, H, freq="h", X_df=fut, explanations=small)
    pd.testing.assert_frame_equal(a, b)
    assert max(c["unique_id"].nunique() for c in model.calls) == 3


def test_warns_but_still_runs_when_estimate_is_high():
    df, _ = make_data(2)
    cfg = ShapExplanations(history=[6, 24], method="exact", warn_model_calls=1)
    with pytest.warns(UserWarning, match="series"):
        out = FakeForecaster().explain(
            df[["unique_id", "ds", "y"]], H, freq="h", explanations=cfg
        )
    assert len(out) == 2 * H


def test_exact_with_many_players_raises_instead_of_exhausting_memory():
    df, _ = make_data(1)
    cfg = ShapExplanations(history=list(range(1, 22)), method="exact")
    with pytest.raises(ValueError, match="permutation"):
        FakeForecaster().explain(
            df[["unique_id", "ds", "y"]], H, freq="h", explanations=cfg
        )


def test_history_boundaries_must_increase():
    df, _ = make_data(1)
    with pytest.raises(ValueError, match="increasing"):
        FakeForecaster().explain(
            df[["unique_id", "ds", "y"]],
            H,
            freq="h",
            explanations=ShapExplanations(history=[24, 6]),
        )


def test_model_without_native_exog_rejects_x_df():
    df, fut = make_data(1)
    with pytest.raises(ValueError, match="exog"):
        FakeForecaster(exog=False).explain(df, H, freq="h", X_df=fut)


def test_model_config_is_default_and_call_overrides():
    df, _ = make_data(1)
    data = df[["unique_id", "ds", "y"]]
    model = FakeForecaster(explanations=ShapExplanations(history=[6], method="exact"))
    default = model.explain(data, H, freq="h")
    assert "Fake-shap-hist-6" in default.columns
    override = model.explain(
        data, H, freq="h", explanations=ShapExplanations(history=[12], method="exact")
    )
    assert "Fake-shap-hist-12" in override.columns


def test_forecast_is_not_affected_by_explanations():
    df, _ = make_data(1)
    data = df[["unique_id", "ds", "y"]]
    plain = FakeForecaster()
    configured = FakeForecaster(
        explanations=ShapExplanations(history=[6], method="exact")
    )
    a = plain.forecast(data, H, freq="h")
    b = configured.forecast(data, H, freq="h")
    pd.testing.assert_frame_equal(a, b)
    assert len(configured.calls) == 1


def test_empty_history_window_gets_no_credit():
    df, _ = make_data(1)
    short = df.tail(40)
    model = FakeForecaster()
    cfg = ShapExplanations(history=[30, 60], exog=False, method="exact")
    out = model.explain(short, H, freq="h", explanations=cfg)

    assert shap_cols(out) == [
        "Fake-shap-base",
        "Fake-shap-hist-30",
        "Fake-shap-hist-30-60",
        "Fake-shap-hist-older",
    ]
    np.testing.assert_allclose(out["Fake-shap-hist-older"], 0.0, atol=1e-10)
    np.testing.assert_allclose(out[shap_cols(out)].sum(axis=1), out["Fake"])
    assert not any(c["y"].isna().all() for c in model.calls)


def test_window_empty_for_only_some_series_gets_no_credit_there():
    df, _ = make_data(2)
    df = pd.concat(
        [df[df["unique_id"] == "s0"], df[df["unique_id"] == "s1"].tail(20)],
        ignore_index=True,
    )
    model = FakeForecaster()
    cfg = ShapExplanations(history=[6, 24], exog=False, method="exact")
    out = model.explain(df, H, freq="h", explanations=cfg)

    short = out[out["unique_id"] == "s1"]
    np.testing.assert_allclose(short["Fake-shap-hist-older"], 0.0, atol=1e-10)
    long = out[out["unique_id"] == "s0"]
    assert (long["Fake-shap-hist-older"].abs() > 1e-6).any()
    np.testing.assert_allclose(out[shap_cols(out)].sum(axis=1), out["Fake"])
