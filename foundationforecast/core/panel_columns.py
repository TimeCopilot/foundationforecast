from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import pandas as pd

CANONICAL_ID_COL = "unique_id"
CANONICAL_TIME_COL = "ds"
CANONICAL_TARGET_COL = "y"


@dataclass(frozen=True)
class PanelColumns:
    """User-facing panel id, time, and target column names."""

    id_col: str = CANONICAL_ID_COL
    time_col: str = CANONICAL_TIME_COL
    target_col: str = CANONICAL_TARGET_COL

    def is_canonical(self) -> bool:
        return (
            self.id_col == CANONICAL_ID_COL
            and self.time_col == CANONICAL_TIME_COL
            and self.target_col == CANONICAL_TARGET_COL
        )

    def validate_id_time_present(self, df: pd.DataFrame) -> None:
        missing = [c for c in (self.id_col, self.time_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input df is missing required columns: {missing}. "
                f"Expected id_col={self.id_col!r}, time_col={self.time_col!r}."
            )

    def validate_present(self, df: pd.DataFrame) -> None:
        self.validate_id_time_present(df)
        if self.target_col not in df.columns:
            raise ValueError(
                f"Input df is missing required columns: [{self.target_col!r}]. "
                f"Expected target_col={self.target_col!r}."
            )

    def _rename_map_to_canonical(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for user, canon in (
            (self.id_col, CANONICAL_ID_COL),
            (self.time_col, CANONICAL_TIME_COL),
            (self.target_col, CANONICAL_TARGET_COL),
        ):
            if user != canon:
                mapping[user] = canon
        return mapping

    def _rename_map_from_canonical(self) -> dict[str, str]:
        return {v: k for k, v in self._rename_map_to_canonical().items()}

    def to_canonical(
        self,
        df: pd.DataFrame | None,
        *,
        require_target: bool = True,
    ) -> pd.DataFrame | None:
        if df is None:
            return None
        mapping = self._rename_map_to_canonical()
        if not require_target:
            mapping = {
                k: v
                for k, v in mapping.items()
                if k in (self.id_col, self.time_col) and k in df.columns
            }
        if not mapping:
            if require_target:
                self.validate_present(df)
            else:
                self.validate_id_time_present(df)
            return df
        if require_target:
            self.validate_present(df)
        else:
            self.validate_id_time_present(df)
        for canon in mapping.values():
            if canon in df.columns and canon not in mapping:
                raise ValueError(
                    f"Column {canon!r} already exists; cannot rename "
                    f"{next(k for k, v in mapping.items() if v == canon)!r}."
                )
        return df.rename(columns=mapping)

    def to_canonical_horizon(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        return self.to_canonical(df, require_target=False)

    def from_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        mapping = self._rename_map_from_canonical()
        if not mapping:
            return df
        return df.rename(columns=mapping)

    def merge_keys(self) -> list[str]:
        return [self.id_col, self.time_col]

    def cv_merge_keys(self) -> list[str]:
        return [self.id_col, self.time_col, "cutoff"]

    def cv_output_prefix(self) -> list[str]:
        return [self.id_col, self.time_col, "cutoff", self.target_col]


def resolve_panel_columns(
    forecaster: object,
    id_col: str | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
) -> PanelColumns:
    defaults = PanelColumns(
        id_col=getattr(forecaster, "id_col", CANONICAL_ID_COL),
        time_col=getattr(forecaster, "time_col", CANONICAL_TIME_COL),
        target_col=getattr(forecaster, "target_col", CANONICAL_TARGET_COL),
    )
    return PanelColumns(
        id_col=id_col if id_col is not None else defaults.id_col,
        time_col=time_col if time_col is not None else defaults.time_col,
        target_col=target_col if target_col is not None else defaults.target_col,
    )


_active_panel_columns_ctx: ContextVar[PanelColumns | None] = ContextVar(
    "active_panel_columns", default=None
)


def get_active_panel_columns() -> PanelColumns | None:
    return _active_panel_columns_ctx.get()


def canonicalize_horizon_exog_frames(
    cols: PanelColumns,
    X_df: pd.DataFrame | None,
    futr_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if X_df is not None and futr_df is not None and X_df is futr_df:
        canonical = cols.to_canonical_horizon(X_df)
        return canonical, canonical
    return cols.to_canonical_horizon(X_df), cols.to_canonical_horizon(futr_df)


@contextmanager
def active_panel_columns(
    _forecaster: object,
    cols: PanelColumns,
):
    token = _active_panel_columns_ctx.set(cols)
    try:
        yield cols
    finally:
        _active_panel_columns_ctx.reset(token)
