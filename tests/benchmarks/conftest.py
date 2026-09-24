import pytest
from tests.helpers import generate_series


@pytest.fixture(scope="session")
def panel_df():
    return generate_series(
        n_series=10,
        freq="D",
        min_length=100,
        max_length=100,
    )


@pytest.fixture(scope="session")
def large_panel_df():
    return generate_series(
        n_series=100,
        freq="D",
        min_length=500,
        max_length=500,
    )


@pytest.fixture(scope="session")
def large_panel_data(large_panel_df):
    from foundationforecast.core.utils import process_panel_from_df

    return process_panel_from_df(large_panel_df)


@pytest.fixture(scope="session")
def string_ds_large_panel_df(large_panel_df):
    df = large_panel_df.copy()
    df["ds"] = df["ds"].astype(str)
    return df


@pytest.fixture(scope="session")
def chronos_bolt():
    from foundationforecast.models.chronos import Chronos

    return Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt")


@pytest.fixture(scope="session")
def timesfm():
    from foundationforecast.models.timesfm import TimesFM

    return TimesFM(
        repo_id="google/timesfm-1.0-200m-pytorch",
        context_length=256,
    )


@pytest.fixture(scope="session")
def toto():
    from foundationforecast.models.toto import Toto

    return Toto(context_length=256, batch_size=2)


@pytest.fixture(scope="session")
def moirai():
    from foundationforecast.models.moirai import Moirai

    return Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-1.1-R-small",
    )
