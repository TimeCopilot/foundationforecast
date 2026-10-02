import pandas as pd
import pytest

from foundationforecast.core.panel_columns import (
    PanelColumns,
    canonicalize_horizon_exog_frames,
)
from foundationforecast.models.tafsut import Tafsut


def _custom_panel():
    df = pd.DataFrame(
        {
            "series_id": ["A", "A", "B", "B"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return df


def test_forecast_custom_column_names():
    df = _custom_panel()
    model = Tafsut(batch_size=2)
    fcst = model.forecast(
        df=df,
        h=2,
        freq="D",
        id_col="series_id",
        time_col="timestamp",
        target_col="target",
    )
    assert list(fcst.columns[:2]) == ["series_id", "timestamp"]
    assert fcst.shape[0] == 4
    assert model.alias in fcst.columns
    assert fcst[model.alias].notna().all()


def test_validate_missing_columns():
    df = pd.DataFrame({"series_id": ["A"], "timestamp": [pd.Timestamp("2024-01-01")]})
    model = Tafsut()
    with pytest.raises(ValueError, match="missing required columns"):
        model.forecast(
            df=df,
            h=1,
            freq="D",
            id_col="series_id",
            time_col="timestamp",
            target_col="target",
        )


def test_horizon_frame_without_target_column():
    cols = PanelColumns(id_col="series_id", time_col="timestamp", target_col="target")
    X_df = pd.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [pd.Timestamp("2024-01-05")],
            "x1": [1.0],
        }
    )
    out = cols.to_canonical_horizon(X_df)
    assert out is not None
    assert list(out.columns) == ["unique_id", "ds", "x1"]


def test_x_df_futr_df_alias_same_object():
    cols = PanelColumns(id_col="series_id", time_col="timestamp", target_col="target")
    frame = pd.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [pd.Timestamp("2024-01-05")],
            "x1": [1.0],
        }
    )
    x, f = canonicalize_horizon_exog_frames(cols, frame, frame)
    assert x is f


def test_tafsut_constructor_panel_column_defaults():
    model = Tafsut(id_col="series_id", time_col="timestamp", target_col="target")
    assert model.id_col == "series_id"
    df = _custom_panel()
    fcst = model.forecast(df=df, h=2, freq="D")
    assert "series_id" in fcst.columns


def test_panel_columns_rename_collision():
    cols = PanelColumns(id_col="unique_id", time_col="ds", target_col="y")
    df = pd.DataFrame(
        {"unique_id": [1], "ds": [pd.Timestamp("2020-01-01")], "y": [1.0], "extra": [0]}
    )
    assert cols.to_canonical(df) is df
