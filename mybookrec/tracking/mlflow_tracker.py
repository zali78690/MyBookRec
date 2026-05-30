"""Thin MLflow wrapper used by training, eval, and backfill scripts.

Goal: one import to start a tracked run, and a clear API for logging params, metrics,
and artifacts. The wrapper centralises tracking-URI + experiment-name resolution so
every entrypoint (live training, backfill, eval reruns) lands in the same backend.

Why a wrapper at all? Two reasons:
- Tests can monkeypatch this module instead of MLflow's stateful globals.
- If we ever swap MLflow for W&B / ClearML, only this file changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from mybookrec.settings import get_settings


def configure_mlflow() -> None:
    """Apply tracking URI + experiment from settings to the global MLflow client.

    Safe to call repeatedly; MLflow ignores re-application of the same URI.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    settings.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(settings.mlflow_experiment_name)


@contextmanager
def track_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run, log static params, and yield the active run.

    Args:
        run_name: Human-readable name shown in the UI.
        params: Hyperparameters and other one-shot key/value pairs.
        tags: Free-form labels (model family, dataset version, etc.).
        nested: True to nest inside an outer run (for sub-experiments).

    Yields:
        The active `mlflow.ActiveRun` object; use `log_metric` / `log_artifact`
        with that run scope.
    """
    configure_mlflow()
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if params:
            log_params(params)
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log one-shot params, stringifying anything MLflow can't serialise directly.

    Args:
        params: Mapping of param name to value. Lists/tuples become JSON-ish strings.
    """
    flat: dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, list | tuple):
            flat[key] = ",".join(str(v) for v in value)
        else:
            flat[key] = str(value)
    mlflow.log_params(flat)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a numeric metric at an optional step (e.g. batch number).

    Args:
        key: Metric name (e.g. "val_hr10", "train_loss").
        value: Float value to record.
        step: X-axis position on the metric chart; omit for one-shot metrics.
    """
    mlflow.log_metric(key, value, step=step)


def log_artifact(path: Path | str, artifact_path: str | None = None) -> None:
    """Upload a file (e.g. a checkpoint or JSONL log) to the run's artifact store.

    Args:
        path: Local file path.
        artifact_path: Subdirectory under the run's artifact root; flat if None.
    """
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
