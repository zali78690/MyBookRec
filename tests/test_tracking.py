"""Tests for the MLflow wrapper + backfill (no real MLflow server needed)."""

from __future__ import annotations

import pathlib

import pytest

from mybookrec.settings import get_settings


def test_track_run_logs_params_and_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Expected use: track_run yields a run; log_metric writes inside the run."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "unit_test")
    get_settings.cache_clear()

    import mlflow

    from mybookrec.tracking import log_metric, track_run

    with track_run(run_name="smoke", params={"foo": 1, "hidden_dims": [512, 256]}):
        log_metric("val_hr10", 0.01, step=1000)
        log_metric("val_hr10", 0.02, step=2000)

    runs = mlflow.search_runs(experiment_names=["unit_test"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.foo"] == "1"
    assert runs.iloc[0]["params.hidden_dims"] == "512,256"
    # The latest val_hr10 logged wins as the headline metric.
    assert runs.iloc[0]["metrics.val_hr10"] == pytest.approx(0.02)
    get_settings.cache_clear()


def test_backfill_summarise_extracts_hr_and_val_loss_series(tmp_path: pathlib.Path) -> None:
    """Expected use: a mixed-event log yields both eval and val_loss series."""
    from mybookrec.tracking.backfill import summarise

    log = [
        {"event": "start", "version": "test_v1", "lr": 1e-3, "dropout": 0.1, "t": 1000.0},
        {"event": "eval", "batch": 5000, "hr10": 0.005, "ndcg10": 0.003, "t": 1100.0},
        {"event": "eval", "batch": 10000, "hr10": 0.012, "ndcg10": 0.007, "t": 1200.0},
        {"event": "val_loss", "batches": 200, "val_loss": 0.32, "t": 1050.0},
        {"event": "final_save", "t": 1300.0},
    ]
    summary = summarise(log)
    assert summary["params"]["version"] == "test_v1"
    assert summary["params"]["lr"] == 1e-3
    assert summary["eval_series"] == [(5000, 0.005, 0.003), (10000, 0.012, 0.007)]
    assert summary["val_loss_series"] == [(200, 0.32)]
    assert summary["best_hr10"] == pytest.approx(0.012)
    assert summary["total_duration_s"] == pytest.approx(300.0)


def test_backfill_summarise_handles_empty_log() -> None:
    """Failure case: log with no eval / val_loss events still returns a safe summary."""
    from mybookrec.tracking.backfill import summarise

    summary = summarise([])
    assert summary["eval_series"] == []
    assert summary["val_loss_series"] == []
    assert summary["best_hr10"] == 0.0
    assert summary["total_duration_s"] == 0.0


def test_backfill_discover_logs_respects_only_filter(tmp_path: pathlib.Path) -> None:
    """Edge case: --only filters down to matching filenames."""
    from mybookrec.tracking.backfill import discover_logs

    (tmp_path / "train_log_a.jsonl").write_text("{}\n")
    (tmp_path / "train_log_b.jsonl").write_text("{}\n")
    (tmp_path / "other.jsonl").write_text("{}\n")
    all_logs = discover_logs(tmp_path, None)
    assert {p.name for p in all_logs} == {"train_log_a.jsonl", "train_log_b.jsonl"}
    only_a = discover_logs(tmp_path, ["train_log_a.jsonl"])
    assert {p.name for p in only_a} == {"train_log_a.jsonl"}


def test_backfill_iter_events_skips_malformed_lines(tmp_path: pathlib.Path) -> None:
    """Failure case: malformed lines are silently skipped, valid ones still yielded."""
    from mybookrec.tracking.backfill import iter_events

    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        '{"event": "start", "t": 1.0}\nnot valid json\n\n{"event": "eval", "batch": 100, "hr10": 0.1, "ndcg10": 0.05}\n'
    )
    events = list(iter_events(log_path))
    assert len(events) == 2
    assert [e["event"] for e in events] == ["start", "eval"]


def test_log_params_stringifies_lists() -> None:
    """Edge case: list/tuple param values become comma-joined strings (MLflow can't take them)."""
    import sys

    fake_calls = []

    def fake_log_params(params: dict) -> None:
        fake_calls.append(params)

    # Stash + restore MLflow's function so we don't need a real server.
    mlflow_mod = sys.modules["mybookrec.tracking.mlflow_tracker"].mlflow
    original = mlflow_mod.log_params
    mlflow_mod.log_params = fake_log_params
    try:
        from mybookrec.tracking import log_params

        log_params({"a": 1, "b": [1, 2, 3], "c": (4, 5)})
    finally:
        mlflow_mod.log_params = original

    assert fake_calls == [{"a": "1", "b": "1,2,3", "c": "4,5"}]
