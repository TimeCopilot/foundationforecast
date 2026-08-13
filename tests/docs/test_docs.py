from pathlib import Path

import pytest
from mktestdocs import check_md_file


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    [p for p in Path("docs").rglob("*.md")],
    ids=str,
)
@pytest.mark.flaky(reruns=3, reruns_delay=80)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
@pytest.mark.flaky(reruns=3, reruns_delay=80)
def test_readme():
    check_md_file("README.md", memory=True)


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    list(Path("foundationforecast").glob("**/*.py")),
    ids=str,
)
def test_py_examples(fpath):
    check_md_file(fpath=fpath, memory=True)
