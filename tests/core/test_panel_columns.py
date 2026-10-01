import pandas as pd
import pytest

from foundationforecast.core.panel_columns import PanelColumns
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


def test_panel_columns_rename_collision():
    cols = PanelColumns(id_col="unique_id", time_col="ds", target_col="y")
    df = pd.DataFrame(
        {"unique_id": [1], "ds": [pd.Timestamp("2020-01-01")], "y": [1.0], "extra": [0]}
    )
    assert cols.to_canonical(df) is df
