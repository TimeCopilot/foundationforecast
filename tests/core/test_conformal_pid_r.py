"""Cross-check PID against R's ``conformalForecast::pid`` (reference fixture).

The fixture is produced by ``fixtures/pid_r/generate.py`` (R 4.4, conformalForecast
0.2.1) from rolling-window AR(2) cross-validation of one series with h=3.
Every interval R emits is the decision made at one forecast origin, so each one
is reproduced here by replaying only the windows known at that origin.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from foundationforecast.core.conformal_pid import track_quantiles

FIXTURE = Path(__file__).parent / "fixtures" / "pid_r" / "pid_reference.json"


def _matrix(rows) -> np.ndarray:
    return np.array([[np.nan if v == "NA" else v for v in row] for row in rows])


@pytest.fixture(scope="module")
def reference():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(
    "name",
    ["asym_expanding", "asym_rolling", "sym_expanding", "asym_no_integrator"],
)
def test_pid_matches_r_conformal_forecast(reference, name):
    params = reference["params"]
    horizon = params["H"][0]
    alpha = 1 - params["level"][0] / 100
    config = reference["results"][name]
    errors = _matrix(reference["error"])
    mean = _matrix(reference["mean"])
    lower = _matrix(config["lower"])
    upper = _matrix(config["upper"])
    symmetric = config["symmetric"][0]
    # window k, horizon j+1 -> the error of target k+j+1 (a row of R's ERROR matrix)
    windows = np.full((errors.shape[0], horizon), np.nan)
    for j in range(horizon):
        windows[: errors.shape[0] - j, j] = errors[j:, j]
    checked = 0
    for h in range(1, horizon + 1):
        for row in np.flatnonzero(~np.isnan(lower[:, h - 1])):
            # R's interval at `row` is the decision at origin t = row + 1 - h, which
            # has seen the horizon-h errors of windows 0..t-h.
            n_windows = row + 2 - 2 * h
            if n_windows < 1:
                continue
            kwargs = dict(
                step_size=1,
                final_gap=h,
                lr=params["lr"][0],
                ki=np.array([params["KI"][0]]),
                csat=params["Csat"][0],
                ncal=params["ncal"][0],
                rolling=config["rolling"][0],
                integrate=config["integrate"][0],
                warm_start=False,
            )
            window_scores = windows[:n_windows].reshape(n_windows, 1, horizon)
            if symmetric:
                radius = track_quantiles(
                    np.abs(window_scores), np.array([alpha]), **kwargs
                )[0, 0, h - 1]
                got_lo, got_hi = radius, radius
            else:
                got_lo = track_quantiles(
                    -window_scores, np.array([alpha / 2]), **kwargs
                )[0, 0, h - 1]
                got_hi = track_quantiles(
                    window_scores, np.array([alpha / 2]), **kwargs
                )[0, 0, h - 1]
            point = mean[row, h - 1]
            np.testing.assert_allclose(point - got_lo, lower[row, h - 1], rtol=1e-9)
            np.testing.assert_allclose(point + got_hi, upper[row, h - 1], rtol=1e-9)
            checked += 1
    assert checked > 150
