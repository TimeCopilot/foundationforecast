"""Generate R `conformalForecast` PID reference fixtures on Modal.

    uv run --with modal modal run tests/core/fixtures/pid_r/generate.py

Writes ``pid_reference.json`` next to this file. The fixture stores the
rolling-window AR(2) cross-validation errors and point forecasts together with
the intervals R's ``pid()`` computes, so ``tests/core/test_conformal_pid_r.py``
can check our implementation against the reference without R installed.
"""

# ruff: noqa: E501
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent

R_SCRIPT = r"""
suppressMessages({library(forecast); library(conformalForecast); library(jsonlite)})
x <- ts(read.csv("/work/series.csv")$y)
H <- 3
far2 <- function(x, h, level) Arima(x, order = c(2, 0, 0)) |> forecast(h = h, level = level)
fc <- cvforecast(x, forecastfun = far2, h = H, level = 95, forward = TRUE, window = 40)
Tg <- 200; delta <- 0.01
Csat <- 2 / pi * (ceiling(log(Tg) * delta) - 1 / log(Tg))
KI <- 2
configs <- list(
  list(name = "asym_expanding", symmetric = FALSE, rolling = FALSE, integrate = TRUE),
  list(name = "asym_rolling", symmetric = FALSE, rolling = TRUE, integrate = TRUE),
  list(name = "sym_expanding", symmetric = TRUE, rolling = FALSE, integrate = TRUE),
  list(name = "asym_no_integrator", symmetric = FALSE, rolling = FALSE, integrate = FALSE)
)
out <- list(
  error = unclass(fc$ERROR), mean = unclass(fc$MEAN),
  params = list(H = H, ncal = 10, lr = 0.1, KI = KI, Csat = Csat, level = 95),
  results = list()
)
for (cfg in configs) {
  p <- pid(fc, symmetric = cfg$symmetric, ncal = 10, rolling = cfg$rolling,
           integrate = cfg$integrate, scorecast = FALSE, lr = 0.1, KI = KI, Csat = Csat)
  out$results[[cfg$name]] <- c(cfg, list(
    lower = unclass(p$LOWER[[1]]), upper = unclass(p$UPPER[[1]])))
}
write_json(out, "/work/out.json", digits = NA, na = "string")
"""

image = modal.Image.from_registry("rocker/r-ver:4.4.1", add_python="3.12").run_commands(
    'Rscript -e \'options(warn = 2); install.packages(c("forecast", "conformalForecast", "jsonlite"), repos = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"); library(conformalForecast)\''
)
app = modal.App("pid-r-reference", image=image)


@app.function(timeout=1800)
def run_r(series_csv: str) -> str:
    import subprocess

    Path("/work").mkdir(exist_ok=True)
    Path("/work/series.csv").write_text(series_csv)
    Path("/work/run.R").write_text(R_SCRIPT)
    subprocess.run(["Rscript", "/work/run.R"], check=True)
    return Path("/work/out.json").read_text()


@app.local_entrypoint()
def main():
    import numpy as np

    rng = np.random.default_rng(2024)
    n = 120
    y = np.zeros(n)
    eps = rng.normal(size=n) * np.where(np.arange(n) < 70, 1.0, 2.0)
    for t in range(2, n):
        y[t] = 0.8 * y[t - 1] - 0.5 * y[t - 2] + eps[t]
    csv = "y\n" + "\n".join(f"{v:.17g}" for v in y)
    out = json.loads(run_r.remote(csv))
    out["series"] = y.tolist()
    (HERE / "pid_reference.json").write_text(json.dumps(out))
    print("wrote", HERE / "pid_reference.json")
