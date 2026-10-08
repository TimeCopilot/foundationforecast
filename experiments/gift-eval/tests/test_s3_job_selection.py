from src.eval.jobs import parse_job_dir_suffix


def test_parse_job_dir_suffix_simple_dataset() -> None:
    assert parse_job_dir_suffix("amazon--chronos-2/m4_weekly/short") == (
        "amazon--chronos-2",
        "m4_weekly",
        "short",
    )


def test_parse_job_dir_suffix_slash_in_dataset() -> None:
    assert parse_job_dir_suffix("amazon--chronos-2/electricity/15T/short") == (
        "amazon--chronos-2",
        "electricity/15T",
        "short",
    )


def test_parse_job_dir_suffix_too_short() -> None:
    assert parse_job_dir_suffix("only/two") is None
