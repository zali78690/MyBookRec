"""Shared pytest fixtures.

Most fixtures revolve around an isolated `data_dir` so tests never touch the real
project data tree. We re-instantiate Settings inside each fixture rather than
mutating the cached singleton, because Pydantic Settings reads env vars at
construction time and we want each test to see a clean slate.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest

from mybookrec.settings import Settings, get_settings


@pytest.fixture
def isolated_data_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """Redirect data_dir to a tmp path for one test, clearing the Settings cache.

    Args:
        tmp_path: Pytest's per-test tmp directory.
        monkeypatch: Pytest monkeypatch fixture.

    Yields:
        The tmp path being used as data_dir.
    """
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Settings:
    """Return a Settings instance bound to a tmp data_dir without an .env file.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest's per-test tmp directory.

    Returns:
        A fresh Settings instance with data_dir set to tmp_path.
    """
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GOOGLE_BOOKS_API_KEY", raising=False)
    get_settings.cache_clear()
    return get_settings()
