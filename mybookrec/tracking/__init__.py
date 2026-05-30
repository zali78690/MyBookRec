"""Experiment tracking via MLflow.

Tiny wrapper around MLflow so the rest of the codebase doesn't import it directly.
One context manager (`track_run`) opens a run, logs config + system info, and lets
callers stream metrics. Tracking URI + experiment name come from
`mybookrec.settings`. The default backend is a local SQLite file at `mlruns.db`,
so no server needs to be running.
"""

from mybookrec.tracking.mlflow_tracker import log_artifact, log_metric, log_params, track_run

__all__ = ["track_run", "log_metric", "log_params", "log_artifact"]
