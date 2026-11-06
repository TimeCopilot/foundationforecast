from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Day, Tick

if TYPE_CHECKING:
    from .forecaster import Forecaster

logger = logging.getLogger(__name__)

ShapMethod = Literal["auto", "exact", "permutation", "loo"]
MaskMode = Literal["auto", "nan", "fill"]
HistorySpec = int | str

_METHODS = ("auto", "exact", "permutation", "loo")
_MASK_MODES = ("auto", "nan", "fill")
_EXACT_WARN_PLAYERS = 12
_EXACT_MAX_PLAYERS = 20
_NS_PER_DAY = 86_400 * 10**9


@dataclass
class ShapExplanations:
    """Settings for masking-based Shapley explanations of a forecast.

    The forecast for every series is split into additive contributions from
    *players*: windows of the target history (newest to oldest) and known-future
    exogenous columns. A player is switched off by masking its inputs, so no
    background dataset is sampled. See the explainability guide.

    Args:
        history: Where to cut the history into windows, measured back from the
            last observation. Each item is a number of steps (``int``) or a pandas
            offset string (``"1D"``, ``"7D"``). ``["1D", "7D"]`` gives the windows
            last 1D, 1D to 7D, and everything older. ``None`` (default) picks
            windows from the frequency.
        exog: Which exogenous columns are players. ``True`` (default) uses every
            column in ``X_df``, ``False`` uses none, and a list of names uses
            only those. Columns that are not players are always provided to the
            model.
        method: ``"auto"`` (default) uses ``"exact"`` when it needs no more model
            runs than ``"permutation"`` would, and ``"permutation"`` otherwise.
            ``"exact"`` evaluates all ``2 ** n`` coalitions. ``"permutation"``
            samples permutations. ``"loo"`` removes one player at a time; it is
            cheap but not exactly additive, so a ``-shap-residual`` column is
            added.
        n_samples: Number of permutations for ``"permutation"``. It also sets the
            run budget that ``"auto"`` compares ``"exact"`` against.
        mask: How to switch off a history window that is not the oldest one.
            ``"nan"`` sets its target values to NaN, ``"fill"`` replaces them
            with the per-series baseline, and ``"auto"`` (default) uses ``"nan"``
            for models that accept missing targets and ``"fill"`` otherwise. The
            oldest windows are always removed from the start of the context.
        base_window: History used for the per-series baseline (the value of the
            forecast with every player off): a number of steps or an offset
            string. ``None`` (default) uses the whole series. The baseline is the
            mean of the target over this window.
        warn_model_calls: Emit a ``UserWarning`` when the estimated number of
            series forecasts exceeds this value. The run continues.
        max_batch_series: Maximum number of series stacked into one forecast
            call, which bounds memory on large panels.
        seed: Random seed for ``"permutation"``.
    """

    history: HistorySpec | Sequence[HistorySpec] | None = None
    exog: bool | Sequence[str] = True
    method: ShapMethod = "auto"
    n_samples: int = 10
    mask: MaskMode = "auto"
    base_window: HistorySpec | None = None
    warn_model_calls: int = 2_000
    max_batch_series: int = 2_048
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if self.mask not in _MASK_MODES:
            raise ValueError(f"mask must be one of {_MASK_MODES}")
        if self.n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if self.warn_model_calls < 1:
            raise ValueError("warn_model_calls must be a positive integer")
        if self.max_batch_series < 1:
            raise ValueError("max_batch_series must be a positive integer")
        if self.history is not None:
            specs = _normalize_history(self.history)
            if not specs:
                raise ValueError("history must contain at least one window boundary")
            for spec in specs:
                _validate_spec(spec, "history")
        if self.base_window is not None:
            _validate_spec(self.base_window, "base_window")
        if isinstance(self.exog, str):
            raise ValueError("exog must be a bool or a list of column names")


def _validate_spec(spec: HistorySpec, name: str) -> None:
    if isinstance(spec, bool) or not isinstance(spec, int | np.integer | str):
        raise ValueError(f"{name} entries must be int steps or offset strings.")
    if isinstance(spec, int | np.integer):
        if spec < 1:
            raise ValueError(f"{name} step counts must be positive integers.")
        return
    try:
        to_offset(spec)
    except ValueError as e:
        raise ValueError(f"Invalid offset {spec!r} in {name}: {e}") from e


def _normalize_history(
    history: HistorySpec | Sequence[HistorySpec],
) -> list[HistorySpec]:
    if isinstance(history, str | int | np.integer):
        return [history]
    return list(history)


def _import_shap():
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "Shapley explanations with method='exact' or 'permutation' need the "
            "`shap` package. Install it with `uv add "
            "'foundationforecast[explain]'` (or `pip install "
            "'foundationforecast[explain]'`) or use method='loo'."
        ) from e
    return shap


def auto_history(freq: str, h: int) -> list[HistorySpec]:
    """Default window boundaries for a pandas frequency."""
    offset = to_offset(freq)
    if isinstance(offset, Day):
        return ["7D", "28D"] if offset.n == 1 else [7, 28]
    if isinstance(offset, Tick):
        return ["1D", "7D"] if offset.nanos < _NS_PER_DAY else [7, 28]
    code = offset.rule_code.upper()
    if code.startswith("W"):
        return [4, 13]
    if code.startswith(("Q", "BQ")):
        return [4, 12]
    if code.startswith(("M", "BM", "SM", "CBM")):
        return [3, 12]
    if code.startswith(("Y", "A", "BY", "BA")):
        return [3]
    if code.startswith(("B", "CB")):
        return [5, 20]
    return [h, 4 * h]


def _step_counts(
    ds: pd.Series,
    codes: np.ndarray,
    n_rows: np.ndarray,
    spec: HistorySpec,
) -> np.ndarray:
    """Number of trailing observations covered by ``spec`` for every series."""
    if isinstance(spec, int | np.integer):
        return np.minimum(int(spec), n_rows)
    last = ds.groupby(codes).transform("max")
    inside = (ds > (last - to_offset(spec))).to_numpy().astype(float)
    return np.bincount(codes, weights=inside, minlength=len(n_rows)).astype(int)


def estimate_series_forecasts(
    method: str,
    n_hist: int,
    n_exog: int,
    n_series: int,
    n_samples: int,
) -> int:
    """Upper-bound estimate of series forecasts needed by an explanation."""
    n = n_hist + n_exog
    if method == "exact":
        runs = 2**n - 2**n_exog
    elif method == "permutation":
        runs = 2 * n * n_samples + 1
    else:
        runs = n + 1
    return int(max(runs, 1) * n_series)


def resolve_method(method: str, n_players: int, n_samples: int) -> str:
    """Resolve ``"auto"``: exact when it is no costlier than permutation."""
    if method != "auto":
        return method
    permutation_budget = 2 * n_players * n_samples + 1
    return "exact" if 2**n_players <= permutation_budget else "permutation"


class CoalitionEvaluator:
    """Callable ``f(masks) -> forecasts`` over binary player masks.

    Rows of ``masks`` are coalitions (1 = player on, history players first, then
    exogenous players). Each coalition becomes a masked copy of the panel, and
    all copies that use the same exogenous columns go through one stacked
    ``forecast`` call. Coalitions with no history window evaluate to the
    baseline. Results are memoised per coalition.
    """

    def __init__(
        self,
        forecaster: Forecaster,
        df: pd.DataFrame,
        h: int,
        freq: str,
        horizon_df: pd.DataFrame | None,
        exog_players: list[str],
        fixed_exog: list[str],
        windows: np.ndarray,
        n_hist: int,
        codes: np.ndarray,
        series_index: pd.Index,
        baseline: np.ndarray,
        use_nan: bool,
        max_batch_series: int,
        context_length: int | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.h = h
        self.freq = freq
        self.exog_players = exog_players
        self.fixed_exog = fixed_exog
        self.windows = windows
        self.n_hist = n_hist
        self.codes = codes
        # (series, window) pairs the model can see: the window holds at least
        # one observation and, when the model caps its context, at least one of
        # them falls inside it. A window the model cannot see never counts as
        # "on": switching it on or off cannot change the model input.
        pos_from_end = pd.Series(codes).groupby(codes).cumcount(ascending=False)
        visible = (
            np.ones(len(codes), dtype=bool)
            if context_length is None
            else pos_from_end.to_numpy() < context_length
        )
        n_series = len(series_index)
        self.nonempty = (
            np.bincount(
                codes[visible] * n_hist + windows[visible],
                minlength=n_series * n_hist,
            )
            .reshape(n_series, n_hist)
            .astype(bool)
        )
        self.always_dead = ~self.nonempty.any(axis=0)
        self.n_series = n_series
        self.baseline = baseline
        self.use_nan = use_nan
        self.max_batch_series = max_batch_series
        self.n_out = self.n_series * h
        self.base_flat = np.repeat(baseline, h)
        self._ds = df["ds"].to_numpy()
        self._y = df["y"].to_numpy(dtype=float)
        self._exog_hist = {c: df[c].to_numpy() for c in [*fixed_exog, *exog_players]}
        self._horizon = horizon_df
        self._x_codes = (
            series_index.get_indexer(horizon_df["unique_id"])
            if horizon_df is not None
            else None
        )
        self.future_ds: np.ndarray | None = None
        self.n_series_forecasts = 0
        self._cache: dict[bytes, np.ndarray] = {}

    def _exog_cols(self, row: np.ndarray) -> tuple[str, ...]:
        on = [
            name
            for name, flag in zip(self.exog_players, row[self.n_hist :], strict=True)
            if flag
        ]
        return (*self.fixed_exog, *on)

    def _masked_history(
        self,
        hist_on: np.ndarray,
        cols: tuple[str, ...],
        labels: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Masked panel for one coalition and the series that fall back to base.

        Empty windows are ignored per series. A series whose non-empty windows
        are all off is explained by the baseline, so it is sent through the
        model with its newest window unmasked and its forecast is replaced by
        the baseline afterwards.
        """
        eff_on = self.nonempty & hist_on[None, :]
        dead = ~eff_on.any(axis=1)
        eff_on[dead, 0] = True
        positions = np.arange(self.n_hist)
        oldest_on = np.where(eff_on, positions[None, :], -1).max(axis=1)
        keep = self.windows <= oldest_on[self.codes]
        masked = keep & ~eff_on[self.codes, self.windows]
        y = self._y.copy()
        if self.use_nan:
            y[masked] = np.nan
        else:
            y[masked] = self.baseline[self.codes][masked]
        idx = np.flatnonzero(keep)
        out = {
            "unique_id": labels[self.codes[idx]],
            "ds": self._ds[idx],
            "y": y[idx],
        }
        for c in cols:
            out[c] = self._exog_hist[c][idx]
        return out, dead

    def _stacked_horizon(
        self,
        cols: tuple[str, ...],
        labels: np.ndarray,
    ) -> pd.DataFrame:
        assert self._horizon is not None and self._x_codes is not None
        return pd.DataFrame(
            {
                "unique_id": labels[self._x_codes],
                "ds": self._horizon["ds"].to_numpy(),
                **{c: self._horizon[c].to_numpy() for c in cols},
            }
        )

    def _run_group(
        self,
        rows: list[int],
        masks: np.ndarray,
        cols: tuple[str, ...],
    ) -> np.ndarray:
        result = np.empty((len(rows), self.n_out))
        step = max(1, self.max_batch_series // self.n_series)
        alias = self.forecaster.alias
        for start in range(0, len(rows), step):
            chunk = rows[start : start + step]
            label_to_pos: dict[str, int] = {}
            hist_parts = []
            horizon_parts = []
            dead_rows = []
            for j, row in enumerate(chunk):
                labels = np.array(
                    [f"c{j}:{s}" for s in range(self.n_series)], dtype=object
                )
                for s, label in enumerate(labels):
                    label_to_pos[label] = j * self.n_series + s
                hist_on = masks[row, : self.n_hist]
                part, dead = self._masked_history(hist_on, cols, labels)
                hist_parts.append(part)
                dead_rows.append(dead)
                if cols:
                    horizon_parts.append(self._stacked_horizon(cols, labels))
            hist_df = pd.DataFrame(
                {k: np.concatenate([p[k] for p in hist_parts]) for k in hist_parts[0]}
            )
            kwargs: dict = {}
            if cols:
                kwargs["X_df"] = pd.concat(horizon_parts, ignore_index=True)
                kwargs["futr_exog_list"] = list(cols)
            fcst = self.forecaster.forecast(hist_df, self.h, freq=self.freq, **kwargs)
            self.n_series_forecasts += len(chunk) * self.n_series
            pos = fcst["unique_id"].map(label_to_pos)
            expected = len(chunk) * self.n_series * self.h
            if len(fcst) != expected or pos.isna().any():
                raise RuntimeError(
                    "Unexpected forecast shape while explaining: got "
                    f"{len(fcst)} rows, expected {expected}."
                )
            order = np.lexsort((fcst["ds"].to_numpy(), pos.to_numpy()))
            vals = fcst[alias].to_numpy(dtype=float)[order]
            block = vals.reshape(len(chunk), self.n_series, self.h).copy()
            for j, dead in enumerate(dead_rows):
                block[j, dead] = self.baseline[dead, None]
            result[start : start + len(chunk)] = block.reshape(len(chunk), self.n_out)
            if self.future_ds is None:
                ds_sorted = fcst["ds"].to_numpy()[order]
                self.future_ds = ds_sorted.reshape(len(chunk), self.n_series, self.h)[0]
        return result

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(np.asarray(masks)) > 0.5
        masks = masks.copy()
        # Windows no series can see are equivalent to "off": canonicalise so
        # coalitions that differ only in them share one model run.
        masks[:, : self.n_hist] &= ~self.always_dead
        keys = [row.tobytes() for row in masks]
        todo: dict[bytes, int] = {}
        for i, (row, key) in enumerate(zip(masks, keys, strict=True)):
            if key in self._cache or key in todo:
                continue
            if not row[: self.n_hist].any():
                self._cache[key] = self.base_flat
            else:
                todo[key] = i
        groups: dict[tuple[str, ...], list[int]] = {}
        for i in todo.values():
            groups.setdefault(self._exog_cols(masks[i]), []).append(i)
        for cols, rows in groups.items():
            values = self._run_group(rows, masks, cols)
            for value, i in zip(values, rows, strict=True):
                self._cache[keys[i]] = value
        return np.stack([self._cache[k] for k in keys])


def _context_limit(forecaster: Forecaster) -> int | None:
    """Number of trailing observations the model reads, when it exposes a cap."""
    limit = getattr(forecaster, "context_length", None)
    if isinstance(limit, int | np.integer) and not isinstance(limit, bool):
        return int(limit) if limit > 0 else None
    return None


def explain_forecaster(
    forecaster: Forecaster,
    df: pd.DataFrame,
    h: int,
    freq: str | None = None,
    X_df: pd.DataFrame | None = None,
    *,
    explanations: ShapExplanations | None = None,
    futr_df: pd.DataFrame | None = None,
    futr_exog_list: list[str] | None = None,
    ids: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Shapley decomposition of ``forecaster.forecast`` (see ``Forecaster.explain``)."""
    from .exog.covariates import resolve_horizon_exog_df
    from .exog.futr_exog import prepare_futr_exog_context
    from .forecaster import Forecaster as _Forecaster
    from .forecaster import maybe_convert_col_to_datetime

    _Forecaster.validate_input(df, h)
    cfg = (
        explanations or getattr(forecaster, "explanations", None) or ShapExplanations()
    )
    freq = forecaster._maybe_infer_freq(df, freq)
    df = maybe_convert_col_to_datetime(df, "ds")
    horizon_df = resolve_horizon_exog_df(X_df, futr_df)
    if horizon_df is not None:
        horizon_df = maybe_convert_col_to_datetime(horizon_df, "ds")
    if ids is not None:
        ids = list(ids)
        present = set(df["unique_id"].unique())
        missing = [i for i in ids if i not in present]
        if missing:
            raise ValueError(f"ids not found in df: {missing}")
        df = df[df["unique_id"].isin(ids)]
        if horizon_df is not None:
            horizon_df = horizon_df[horizon_df["unique_id"].isin(ids)]
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    ctx = prepare_futr_exog_context(
        forecaster,
        df,
        h,
        X_df=horizon_df,
        futr_df=None,
        futr_exog_list=futr_exog_list,
    )
    all_exog = ctx.futr_exog_list if ctx is not None else []
    horizon_df = ctx.horizon_df if ctx is not None else None
    if cfg.exog is True:
        exog_players = list(all_exog)
    elif cfg.exog is False:
        exog_players = []
    else:
        exog_players = list(cfg.exog)
        unknown = [c for c in exog_players if c not in all_exog]
        if unknown:
            raise ValueError(
                f"exog columns {unknown} are not among the exogenous columns "
                f"in use: {all_exog}."
            )
    fixed_exog = [c for c in all_exog if c not in exog_players]

    codes, uniques = pd.factorize(df["unique_id"])
    series_index = pd.Index(uniques)
    n_series = len(series_index)
    n_rows = np.bincount(codes, minlength=n_series)
    pos_from_end = pd.Series(codes).groupby(codes).cumcount(ascending=False).to_numpy()

    explicit_history = cfg.history is not None
    specs = (
        _normalize_history(cfg.history)  # type: ignore[arg-type]
        if explicit_history
        else auto_history(freq, h)
    )
    cut_cols = [_step_counts(df["ds"], codes, n_rows, spec) for spec in specs]
    keep_specs: list[HistorySpec] = []
    keep_cuts: list[np.ndarray] = []
    for spec, cut in zip(specs, cut_cols, strict=True):
        if not explicit_history:
            redundant = (cut >= n_rows).all() or (
                keep_cuts and (cut == keep_cuts[-1]).all()
            )
            if redundant:
                continue
        elif keep_cuts and (cut < keep_cuts[-1]).any():
            raise ValueError(
                "history boundaries must be increasing; "
                f"{spec!r} reaches back less than {keep_specs[-1]!r}."
            )
        keep_specs.append(spec)
        keep_cuts.append(cut)
    cuts = (
        np.stack(keep_cuts, axis=1) if keep_cuts else np.zeros((n_series, 0), dtype=int)
    )
    windows = (pos_from_end[:, None] >= cuts[codes]).sum(axis=1)
    if keep_specs:
        hist_labels = [
            str(keep_specs[0]),
            *[
                f"{keep_specs[k - 1]}-{keep_specs[k]}"
                for k in range(1, len(keep_specs))
            ],
            "older",
        ]
    else:
        hist_labels = ["all"]
    n_hist = len(hist_labels)

    y = df["y"].to_numpy(dtype=float)
    if cfg.base_window is None:
        in_base = np.ones(len(df), dtype=bool)
    else:
        base_counts = _step_counts(df["ds"], codes, n_rows, cfg.base_window)
        in_base = pos_from_end < base_counts[codes]
    baseline = pd.Series(np.where(in_base, y, np.nan)).groupby(codes).mean().to_numpy()
    if np.isnan(baseline).any():
        raise ValueError("Cannot compute a baseline: a series has no observed values.")

    n_players = n_hist + len(exog_players)
    method = resolve_method(cfg.method, n_players, cfg.n_samples)
    if method == "exact" and n_players > _EXACT_MAX_PLAYERS:
        raise ValueError(
            f"method='exact' would evaluate 2**{n_players} coalitions, which does "
            "not fit in memory. Use method='permutation' or reduce the number "
            "of history windows and exogenous players."
        )
    estimate = estimate_series_forecasts(
        method, n_hist, len(exog_players), n_series, cfg.n_samples
    )
    logger.info(
        "Explaining %s: %d players, method=%s, ~%d series forecasts.",
        forecaster.alias,
        n_players,
        method,
        estimate,
    )
    if estimate > cfg.warn_model_calls or (
        method == "exact" and n_players > _EXACT_WARN_PLAYERS
    ):
        warnings.warn(
            f"Explaining {forecaster.alias} needs about {estimate:,} series "
            f"forecasts ({n_players} players, method={method!r}). Pass `ids=` to "
            "explain fewer series, use fewer history windows or exogenous "
            "players, or use method='permutation' or 'loo'.",
            UserWarning,
            stacklevel=3,
        )

    use_nan = cfg.mask == "nan" or (
        cfg.mask == "auto" and getattr(forecaster, "supports_nan_target", False)
    )
    evaluator = CoalitionEvaluator(
        forecaster=forecaster,
        df=df,
        h=h,
        freq=freq,
        horizon_df=horizon_df,
        exog_players=exog_players,
        fixed_exog=fixed_exog,
        windows=windows,
        n_hist=n_hist,
        codes=codes,
        series_index=series_index,
        baseline=baseline,
        use_nan=use_nan,
        max_batch_series=cfg.max_batch_series,
        context_length=_context_limit(forecaster),
    )
    unseen = [hist_labels[i] for i in np.flatnonzero(evaluator.always_dead)]
    if unseen:
        warnings.warn(
            f"History window(s) {unseen} hold no observations the model can see "
            "(they are empty or lie beyond the model's `context_length`), so "
            "they get zero attribution. Use smaller history boundaries.",
            UserWarning,
            stacklevel=3,
        )
    ones = np.ones((1, n_players))
    full = evaluator(ones)[0]
    base_flat = evaluator.base_flat
    if n_players == 1:
        contrib = (full - base_flat)[None, :]
    elif method == "loo":
        contrib = full[None, :] - evaluator(1.0 - np.eye(n_players))
    else:
        shap = _import_shap()
        masker = shap.maskers.Independent(np.zeros((1, n_players)), max_samples=1)
        if method == "exact":
            explainer = shap.explainers.Exact(evaluator, masker)
            explanation = explainer(ones, max_evals=2**n_players + 1, silent=True)
        else:
            explainer = shap.explainers.Permutation(evaluator, masker, seed=cfg.seed)
            explanation = explainer(
                ones, max_evals=2 * n_players * cfg.n_samples + 1, silent=True
            )
        contrib = np.asarray(explanation.values).reshape(n_players, -1)

    assert evaluator.future_ds is not None
    alias = forecaster.alias
    out = pd.DataFrame(
        {
            "unique_id": np.repeat(series_index.to_numpy(), h),
            "ds": evaluator.future_ds.reshape(-1),
            alias: full,
            f"{alias}-shap-base": base_flat,
        }
    )
    names = [f"{alias}-shap-hist-{label}" for label in hist_labels]
    names += [f"{alias}-shap-{c}" for c in exog_players]
    for name, values in zip(names, contrib, strict=True):
        out[name] = values
    if method == "loo":
        out[f"{alias}-shap-residual"] = full - base_flat - contrib.sum(axis=0)
    return out
