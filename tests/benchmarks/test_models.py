import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.models]


@pytest.mark.parametrize(
    "model_fixture,expected_alias",
    [
        pytest.param("chronos_bolt", "Chronos-Bolt", id="chronos-bolt"),
        pytest.param("timesfm", "TimesFM", id="timesfm-1"),
        pytest.param("toto", "Toto", id="toto"),
        pytest.param("moirai", "Moirai", id="moirai-1.1"),
    ],
)
def test_model_forecast(benchmark, model_fixture, expected_alias, panel_df, request):
    model = request.getfixturevalue(model_fixture)
    result = benchmark(model.forecast, panel_df, h=12, freq="D")
    assert len(result) == panel_df["unique_id"].nunique() * 12
    assert expected_alias in result.columns
