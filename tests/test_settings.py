"""Tests for the Pydantic Settings layer."""

from __future__ import annotations

import pathlib

import pytest

from mybookrec.settings import Settings, get_settings


def test_settings_defaults_resolve_under_repo_root(fresh_settings: Settings) -> None:
    """data_dir defaults to a path under the repo (or env override)."""
    assert isinstance(fresh_settings.data_dir, pathlib.Path)
    assert fresh_settings.bronze_dir == fresh_settings.data_dir / "bronze"
    assert fresh_settings.silver_dir == fresh_settings.data_dir / "silver"
    assert fresh_settings.gold_dir == fresh_settings.data_dir / "gold"


def test_settings_picks_up_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Env vars take precedence over defaults."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "test-key-xyz")
    monkeypatch.setenv("SERVE_PORT", "9999")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.data_dir == tmp_path
    assert settings.google_books_api_key == "test-key-xyz"
    assert settings.serve_port == 9999
    get_settings.cache_clear()


def test_settings_validates_serve_port_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range ports fail validation."""
    monkeypatch.setenv("SERVE_PORT", "99999")
    get_settings.cache_clear()
    with pytest.raises(Exception):  # pydantic.ValidationError
        Settings()
    get_settings.cache_clear()


def test_resolved_serve_model_path_falls_back(fresh_settings: Settings) -> None:
    """When serve_model_path is None, default to <checkpoints_dir>/two_tower_mac.pt."""
    assert fresh_settings.serve_model_path is None
    fallback = fresh_settings.resolved_serve_model_path()
    assert fallback.name == "two_tower_mac.pt"
    assert fallback.parent == fresh_settings.checkpoints_dir
