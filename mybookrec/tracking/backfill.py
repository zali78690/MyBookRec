"""Replay historical training runs into MLflow from `checkpoints/train_log_*.jsonl`.

Each per-batch JSONL file is one historical run. We reconstruct:

- Run name and tags from filename + `start` event's `version` field.
- Params from the `start` event (time_budget, dropout, weight_decay, negative_sampling).
- Per-eval `val_hr10` / `val_ndcg10` time series from each `eval` event.
- Best HR@10 + total duration as one-shot metrics.
- The original .jsonl as a run artifact, so future inspection has the raw log.

Re-running is idempotent: each backfilled run gets a deterministic name
"<filename-stem>_backfill" so re-imports don't create duplicates (the existing run
with that name is updated, not appended).

Usage:
    .venv/bin/python -m mybookrec.tracking.backfill
    .venv/bin/python -m mybookrec.tracking.backfill --logs-dir checkpoints --only train_log_v4bce.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlflow

from mybookrec.settings import get_settings
from mybookrec.tracking.mlflow_tracker import (
    configure_mlflow,
    log_artifact,
    log_metric,
    log_params,
)

START_EVENT_PARAM_KEYS = (
    "version",
    "time_budget_s",
    "dropout",
    "weight_decay",
    "negative_sampling",
    "lr",
    "batch_size",
    "n_negatives",
    "device",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        Namespace with logs_dir + optional only-filter.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    settings = get_settings()
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=settings.checkpoints_dir,
        help="Directory containing train_log_*.jsonl files.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Process only the given filename(s) (repeatable). Default: all train_log_*.jsonl.",
    )
    return parser.parse_args()


def discover_logs(logs_dir: Path, only: list[str] | None) -> list[Path]:
    """Find training-log JSONLs to backfill.

    Args:
        logs_dir: Directory to scan.
        only: If given, restrict to these filenames.

    Returns:
        Sorted list of matching JSONL paths.
    """
    candidates = sorted(logs_dir.glob("train_log*.jsonl"))
    if only:
        wanted = set(only)
        candidates = [p for p in candidates if p.name in wanted]
    return candidates


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON events from a JSONL file, tolerating malformed lines.

    Args:
        path: Log file to read.

    Yields:
        Each parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract params + eval series + summary metrics from one run's events.

    Args:
        events: All events from one log file.

    Returns:
        Dict with keys: params, eval_series (list of (batch, hr10, ndcg10)),
        best_hr10, total_duration_s, start_event.
    """
    start = next((e for e in events if e.get("event") == "start"), {})
    params = {k: start[k] for k in START_EVENT_PARAM_KEYS if k in start}

    eval_series: list[tuple[int, float, float]] = []
    val_loss_series: list[tuple[int, float]] = []
    for event in events:
        kind = event.get("event")
        if kind == "eval":
            batch = event.get("batch")
            hr10 = event.get("hr10")
            ndcg10 = event.get("ndcg10")
            if batch is not None and hr10 is not None and ndcg10 is not None:
                eval_series.append((int(batch), float(hr10), float(ndcg10)))
        elif kind == "val_loss":
            # v1 schema: {"event":"val_loss","val_loss":..., "batches":...}.
            batch = event.get("batches") or event.get("batch")
            val_loss = event.get("val_loss")
            if batch is not None and val_loss is not None:
                val_loss_series.append((int(batch), float(val_loss)))

    best_hr10 = max((hr for _, hr, _ in eval_series), default=0.0)

    timestamps = [e.get("t") for e in events if isinstance(e.get("t"), int | float)]
    duration_s = float(max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0

    return {
        "params": params,
        "eval_series": eval_series,
        "val_loss_series": val_loss_series,
        "best_hr10": best_hr10,
        "total_duration_s": duration_s,
        "start_event": start,
    }


def find_run_by_name(experiment_id: str, run_name: str) -> str | None:
    """Look up the most recent run with the given name in an experiment.

    Args:
        experiment_id: MLflow experiment id.
        run_name: Run name to match.

    Returns:
        Run id if found, else None.
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def backfill_one(log_path: Path) -> str:
    """Replay one JSONL log into MLflow. Updates the existing run if same name.

    Args:
        log_path: Path to a train_log_*.jsonl file.

    Returns:
        MLflow run name created/updated.
    """
    events = list(iter_events(log_path))
    if not events:
        return ""

    summary = summarise(events)
    stem = log_path.stem
    run_name = f"{stem}_backfill"

    experiment_id = mlflow.get_experiment_by_name(get_settings().mlflow_experiment_name).experiment_id
    existing_run_id = find_run_by_name(experiment_id, run_name)

    with mlflow.start_run(run_id=existing_run_id, run_name=run_name) as _:
        mlflow.set_tags({"backfill": "true", "source_log": log_path.name})
        if summary["params"]:
            log_params(summary["params"])
        for batch, hr10, ndcg10 in summary["eval_series"]:
            log_metric("val_hr10", hr10, step=batch)
            log_metric("val_ndcg10", ndcg10, step=batch)
        for batch, val_loss in summary["val_loss_series"]:
            log_metric("val_loss", val_loss, step=batch)
        log_metric("best_val_hr10", summary["best_hr10"])
        log_metric("total_train_time_s", summary["total_duration_s"])
        log_artifact(log_path, artifact_path="train_logs")
    return run_name


def main() -> None:
    """Discover historical logs and backfill each into MLflow."""
    args = parse_args()
    configure_mlflow()
    logs = discover_logs(args.logs_dir, args.only)
    if not logs:
        print(f"No train_log_*.jsonl files in {args.logs_dir}")
        return
    print(f"Backfilling {len(logs)} run(s) into MLflow at {get_settings().mlflow_tracking_uri}")
    for path in logs:
        run_name = backfill_one(path)
        print(f"  ✓ {path.name} → run '{run_name}'")
    print("Done. Run `mlflow ui` (with MLFLOW_TRACKING_URI exported) to view.")


if __name__ == "__main__":
    main()
