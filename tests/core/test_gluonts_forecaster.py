import pandas as pd

from foundationforecast.core.gluonts_forecaster import (
    fix_freq,
    maybe_convert_col_to_float32,
)


def test_fix_freq_month_start_and_end():
    assert fix_freq("MS") == "M"
    assert fix_freq("ME") == "M"
    assert fix_freq("D") == "D"


def test_maybe_convert_col_to_float32():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    out = maybe_convert_col_to_float32(df, "y")
    assert out["y"].dtype == "float32"
    assert out is not df
