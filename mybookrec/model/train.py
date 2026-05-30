"""Train a two-tower BCE model.

Auto-detects which feature set (v1 vs v4) to use based on what's on disk.
Override via --feature-set if you need to pin to a specific version.

Usage:
    .venv/bin/python -m mybookrec.model.train
    .venv/bin/python -m mybookrec.model.train --time-budget 3600 --feature-set v4
    .venv/bin/python -m mybookrec.model.train --dropout 0.3 --weight-decay 1e-5
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mybookrec import DATA_DIR
from mybookrec.eval.metrics import hit_rate_at_k, ndcg_at_k
from mybookrec.features.training_pairs import TrainingPairsDataset
from mybookrec.io import (
    FEATURE_SETS,
    batch_encode,
    detect_available_feature_set,
    load_features_for_checkpoint,
    sample_test_pairs,
    select_device,
)
from mybookrec.model.towers import TwoTowerModel
from mybookrec.tracking import log_artifact, log_metric, track_run


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training run.

    Returns:
        argparse.Namespace with hyperparameters and run-control flags.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [fs.name for fs in FEATURE_SETS]
    p.add_argument(
        "--feature-set",
        choices=["auto", *names],
        default="auto",
        help="Feature set name. 'auto' picks the latest available on disk.",
    )
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-negatives", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--negative-sampling", choices=["uniform", "log_freq"], default="uniform")
    p.add_argument("--time-budget", type=int, default=5400, help="Seconds (default 90 min).")
    p.add_argument("--eval-every-batches", type=int, default=5000)
    p.add_argument("--patience", type=int, default=5, help="Early-stop after N evals without HR@10 improvement.")
    p.add_argument("--n-eval-pairs", type=int, default=4000)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument(
        "--checkpoint-name", default="two_tower", help="Saved as <name>.pt (rolling) and <name>_best.pt (best HR@10)."
    )
    p.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Override the MLflow run name. Defaults to '<checkpoint_name>_<feature_set>'.",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run (useful for quick local experiments).",
    )
    return p.parse_args()


def make_evaluator(
    model: TwoTowerModel,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    exclude_dict: dict[int, np.ndarray],
    u_arr: np.ndarray,
    u_t: torch.Tensor,
    b_t: torch.Tensor,
) -> Callable[[], tuple[float, float]]:
    """Build a closure that evaluates HR@10 and NDCG@10 on a fixed test sample.

    Args:
        model: The two-tower model under training.
        user_features: Full bulk-user feature matrix on the training device.
        item_features: Full item feature matrix on the training device.
        exclude_dict: Per-user array of train-rated book indices to mask.
        u_arr: Numpy array of test user indices (for exclude dict lookup).
        u_t: Same as u_arr but as a torch tensor on the training device.
        b_t: Test book index torch tensor (held-out positives).

    Returns:
        A no-arg callable returning (hr10, ndcg10).
    """

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            all_item_embs = batch_encode(model.encode_item, item_features)
            user_embs_eval = model.encode_user(user_features[u_t])
            hits, ndcgs = [], []
            for start in range(0, len(u_t), 64):
                end = min(start + 64, len(u_t))
                scores = user_embs_eval[start:end] @ all_item_embs.T
                for i in range(end - start):
                    excluded = exclude_dict.get(int(u_arr[start + i]))
                    if excluded is not None and len(excluded) > 0:
                        scores[i, excluded] = float("-inf")
                targets = b_t[start:end]
                hits.append(hit_rate_at_k(scores, targets, k=10).cpu())
                ndcgs.append(ndcg_at_k(scores, targets, k=10).cpu())
        model.train()
        return torch.cat(hits).float().mean().item(), torch.cat(ndcgs).mean().item()

    return evaluate


def save_checkpoint(
    path: Path,
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    config: dict,
    hr: float | None = None,
    ndcg: float | None = None,
    n_batches: int | None = None,
) -> None:
    """Serialize a checkpoint containing model + optimizer state + config + metrics.

    Args:
        path: Output .pt path.
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save (for resuming).
        config: Hyperparameter dict (used by `io.load_checkpoint` to re-instantiate).
        hr: Optional held-out HR@10 metric at the time of save.
        ndcg: Optional held-out NDCG@10 metric.
        n_batches: How many batches the model has trained for.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_hr10": hr,
            "val_ndcg10": ndcg,
            "n_batches_trained": n_batches,
            "config": config,
        },
        path,
    )


def main() -> None:
    """Train the two-tower model with BCE + per-positive negative sampling."""
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    feature_set = (
        detect_available_feature_set()
        if args.feature_set == "auto"
        else next(fs for fs in FEATURE_SETS if fs.name == args.feature_set)
    )
    user_features, item_features = load_features_for_checkpoint(
        {"item_input_dim": feature_set.item_input_dim},
        device,
    )
    print(f"Feature set: {feature_set.name}  user: {tuple(user_features.shape)}  item: {tuple(item_features.shape)}")

    train_dataset = TrainingPairsDataset(
        n_negatives=args.n_negatives,
        data_split="train",
        verbose=False,
        negative_sampling=args.negative_sampling,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset ready: {len(train_dataset):,} positives, {len(train_dataset.exclude):,} users")

    u_arr, b_arr = sample_test_pairs(args.n_eval_pairs, seed=42)
    u_t = torch.from_numpy(u_arr.copy()).to(device)
    b_t = torch.from_numpy(b_arr.copy()).to(device)
    print(f"Eval sample: {len(u_arr):,} held-out (user, positive) pairs")

    model = TwoTowerModel(
        user_input_dim=user_features.shape[1],
        item_input_dim=item_features.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    config = {
        "feature_set": feature_set.name,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "user_input_dim": user_features.shape[1],
        "item_input_dim": item_features.shape[1],
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_negatives": args.n_negatives,
        "negative_sampling": args.negative_sampling,
        "loss": "bce",
    }
    ckpt_dir = DATA_DIR.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    rolling_path = ckpt_dir / f"{args.checkpoint_name}.pt"
    best_path = ckpt_dir / f"{args.checkpoint_name}_best.pt"

    evaluate = make_evaluator(model, user_features, item_features, train_dataset.exclude, u_arr, u_t, b_t)

    print(
        f"Training for up to {args.time_budget}s, eval every {args.eval_every_batches} batches, "
        f"patience={args.patience}"
    )

    run_name = args.mlflow_run_name or f"{args.checkpoint_name}_{feature_set.name}"
    run_ctx = (
        track_run(run_name=run_name, params={**config, "time_budget_s": args.time_budget})
        if not args.no_mlflow
        else NullRunContext()
    )

    with run_ctx:
        best_hr = train_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            user_features=user_features,
            item_features=item_features,
            device=device,
            evaluate=evaluate,
            config=config,
            args=args,
            rolling_path=rolling_path,
            best_path=best_path,
            tracking_enabled=not args.no_mlflow,
        )

    print(f"\nDone: best HR@10={best_hr:.4f}")
    print(f"Best checkpoint: {best_path}")


class NullRunContext:
    """No-op context manager used when --no-mlflow is set."""

    def __enter__(self) -> None:
        """Enter no-op context."""

    def __exit__(self, *exc: object) -> None:
        """Exit no-op context."""


def train_loop(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    device: str,
    evaluate: Callable[[], tuple[float, float]],
    config: dict,
    args: argparse.Namespace,
    rolling_path: Path,
    best_path: Path,
    tracking_enabled: bool,
) -> float:
    """Run the training loop. Logs per-eval metrics to MLflow if enabled.

    Args:
        model: TwoTowerModel under training.
        optimizer: Optimizer.
        loss_fn: BCEWithLogitsLoss or equivalent.
        train_loader: DataLoader over the training pairs dataset.
        user_features: Full bulk-user feature tensor on the training device.
        item_features: Full item feature tensor on the training device.
        device: Torch device string (only used for tensor moves).
        evaluate: Closure returning (hr10, ndcg10).
        config: Hyperparameter dict (saved into checkpoints).
        args: Parsed CLI args.
        rolling_path: Where to write the rolling (every-N-batches) checkpoint.
        best_path: Where to write the best-HR@10 checkpoint.
        tracking_enabled: If True, push metrics through `mybookrec.tracking`.

    Returns:
        Best HR@10 observed during training.

    Raises:
        RuntimeError: If the loader yields zero batches.
    """
    model.train()
    t_start = time.time()
    losses: list[float] = []
    last_eval = 0
    last_ckpt = 0
    best_hr = 0.0
    evals_no_improvement = 0
    batch_idx = 0
    for batch_idx, (u_idx, p_idx, n_idx) in enumerate(train_loader, 1):
        u_idx = u_idx.to(device)
        item_idx = torch.cat([p_idx.unsqueeze(1), n_idx], dim=1).to(device)
        similarity = model(user_features[u_idx], item_features[item_idx])
        labels = torch.zeros_like(similarity)
        labels[:, 0] = 1.0
        loss = loss_fn(similarity, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if batch_idx % args.print_every == 0:
            elapsed = time.time() - t_start
            recent = sum(losses[-args.print_every :]) / args.print_every
            print(
                f"batch {batch_idx:6d}  loss={recent:.4f}  temp={model.log_temperature.exp().item():.2f}  "
                f"rate={batch_idx / elapsed:.2f} b/s  elapsed={elapsed:.0f}s"
            )
            if tracking_enabled:
                log_metric("train_loss", recent, step=batch_idx)
                log_metric("train_temperature", model.log_temperature.exp().item(), step=batch_idx)

        if batch_idx - last_eval >= args.eval_every_batches:
            hr, ndcg = evaluate()
            print(f"  EVAL batch={batch_idx}  hr@10={hr:.4f}  ndcg@10={ndcg:.4f}  best={best_hr:.4f}")
            if tracking_enabled:
                log_metric("val_hr10", hr, step=batch_idx)
                log_metric("val_ndcg10", ndcg, step=batch_idx)
            last_eval = batch_idx
            if hr > best_hr:
                best_hr = hr
                save_checkpoint(best_path, model, optimizer, config, hr=hr, ndcg=ndcg, n_batches=batch_idx)
                print(f"  ↳ new best, saved {best_path.name}")
                evals_no_improvement = 0
            else:
                evals_no_improvement += 1
                if evals_no_improvement >= args.patience:
                    print(f"  ↳ early stop after {args.patience} non-improvements (best hr@10={best_hr:.4f})")
                    break

        if batch_idx - last_ckpt >= args.checkpoint_every:
            save_checkpoint(rolling_path, model, optimizer, config, n_batches=batch_idx)
            last_ckpt = batch_idx

        if time.time() - t_start > args.time_budget:
            print(f"Time budget reached at batch {batch_idx}")
            break

    if batch_idx == 0:
        raise RuntimeError("Training loop yielded zero batches — check dataset size and drop_last.")
    save_checkpoint(rolling_path, model, optimizer, config, n_batches=batch_idx)
    if tracking_enabled:
        log_metric("best_val_hr10", best_hr)
        log_metric("total_batches", float(batch_idx))
        log_metric("total_train_time_s", time.time() - t_start)
        if best_path.exists():
            log_artifact(best_path, artifact_path="checkpoints")
    return best_hr


if __name__ == "__main__":
    main()
