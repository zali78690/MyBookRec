"""Tests for the rebuild CLI orchestrator."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from mybookrec.features import rebuild


def test_run_builder_sets_env_var_and_invokes_subprocess() -> None:
    """Expected use: run_builder shells out with the requested env."""
    env = {"MYBOOKREC_EMBED_MODEL_RUN": "v2_mxbai", "OTHER": "x"}
    with patch.object(subprocess, "run", return_value=MagicMock(returncode=0)) as mock_run:
        rebuild.run_builder("mybookrec.features.build_user_features", env=env)
    assert mock_run.called
    call = mock_run.call_args
    assert call.args[0] == [sys.executable, "-m", "mybookrec.features.build_user_features"]
    assert call.kwargs["env"] == env


def test_run_builder_propagates_nonzero_exit() -> None:
    """Failure case: a failing builder triggers SystemExit with the same code."""
    with patch.object(subprocess, "run", return_value=MagicMock(returncode=2)), pytest.raises(SystemExit) as exc:
        rebuild.run_builder("mybookrec.features.build_user_features", env={})
    assert exc.value.code == 2


def test_main_runs_three_builders_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expected use: main() shells out the three builders in dependency order."""
    monkeypatch.setattr(sys, "argv", ["rebuild", "--model-run", "v2_mxbai"])
    calls: list[str] = []

    def fake_run(module: str, env: dict[str, str]) -> None:
        calls.append(module)
        assert env["MYBOOKREC_EMBED_MODEL_RUN"] == "v2_mxbai"

    monkeypatch.setattr(rebuild, "run_builder", fake_run)
    rebuild.main()
    assert calls == list(rebuild.BUILDERS)


def test_main_respects_only_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge case: --only restricts the pipeline to the named builder."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["rebuild", "--model-run", "v1_minilm", "--only", "mybookrec.features.build_user_features"],
    )
    calls: list[str] = []
    monkeypatch.setattr(rebuild, "run_builder", lambda module, env: calls.append(module))
    rebuild.main()
    assert calls == ["mybookrec.features.build_user_features"]
