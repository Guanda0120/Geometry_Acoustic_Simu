import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    """Config paths are relative to the repository root, so run every test from there."""
    monkeypatch.chdir(REPO_ROOT)
