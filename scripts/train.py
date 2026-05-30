"""Train a two-tower model.

Single entry point that handles both feature sets:
- v1 (original): user 779-dim, item 395-dim. Loaded from train_user_features.npy + book_embeddings.npy + genre_matrix.npy + num_pages_normalized.npy.
- v4 (with author): user 1163-dim, item 779-dim. Loaded from train_user_features_v4.npy + item_features_v4.npy.

Auto-detects which feature set to use based on what's on disk (prefers v4 if present).

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --time-budget 3600 --batch-size 1024 --feature-set v4
    uv run python scripts/train.py --dropout 0.3 --weight-decay 1e-5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mybookrec import DATA_DIR
from mybookrec.features.training_pairs import TrainingPairsDataset
from mybookrec.model.towers import TwoTowerModel
from mybookrec.eval.metrics import hit_rate_at_k, ndcg_at_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-set", choices=["auto", "v1", "v4"], default="auto",
                   help="v4 (with author features) if available, fallback to v1.")
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
    p.add_argument("--checkpoint-name", default="two_tower",
                   help="Saved as <name>.pt (rolling) and <name>_best.pt (best HR@10).")
    return p.parse_args()


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_features(feature_set: str, device: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Returns (user_features, item_features, resolved_set)."""
    t = DATA_DIR / "transformed"
    v4_user = t / "train_user_features_v4.npy"
    v4_item = t / "item_features_v4.npy"

    if feature_set == "auto":
        feature_set = "v4" if v4_user.exists() and v4_item.exists() else "v1"

    if feature_set == "v4":
        user_features = torch.from_numpy(np.load(v4_user)).to(device).float()
        item_features = torch.from_numpy(np.load(v4_item)).to(device).float()
    else:
        user_features = torch.from_numpy(np.load(t / "train_user_features.npy")).to(device).float()
        _emb = torch.from_numpy(np.load(t / "book_embeddings.npy")).to(device).float()
        _genre = torch.from_numpy(np.load(t / "genre_matrix.npy")).to(device).float()
        _pages = torch.from_numpy(np.load(t / "num_pages_normalized.npy")).to(device).float().reshape(-1, 1)
        item_features = torch.cat([_emb, _genre, _pages], dim=1).contiguous()
        del _emb, _genre, _pages

    return user_features, item_features, feature_set


def build_eval_sample(n_pairs: int, device: str) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Sample N held-out test pairs for periodic HR@10 evaluation."""
    t = DATA_DIR / "transformed"
    with open(t / "user_id_to_index.json") as f:
        user_id_to_index = json.load(f)
    with open(t / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)

    user_map = pl.DataFrame(
        {"user_id": list(user_id_to_index.keys()), "user_idx": list(user_id_to_index.values())},
        schema={"user_id": pl.String, "user_idx": pl.Int64},
    )
    book_map = pl.DataFrame(
        {"book_id": list(book_id_to_index.keys()), "book_idx": list(book_id_to_index.values())},
        schema={"book_id": pl.String, "book_idx": pl.Int64},
    )

    test_df = (
        pl.scan_parquet(t / "training_interactions.parquet")
        .filter(pl.col("data_split") == "test")
        .select("user_id", "book_id", "rating")
        .with_columns(pl.col("book_id").cast(pl.String))
        .join(user_map.lazy(), on="user_id", how="left")
        .join(book_map.lazy(), on="book_id", how="left")
        .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
        .filter(pl.col("rating") >= 4)
        .collect()
    )

    rng = np.random.default_rng(42)
    sample = test_df[rng.choice(len(test_df), size=min(n_pairs, len(test_df)), replace=False)]
    u_arr = sample["user_idx"].to_numpy()
    b_arr = sample["book_idx"].to_numpy()
    return u_arr, b_arr, torch.from_numpy(u_arr.copy()).to(device), torch.from_numpy(b_arr.copy()).to(device)


def make_evaluator(model, user_features, item_features, exclude_dict, u_arr, u_t, b_t):
    """Closure that evaluates HR@10 and NDCG@10 on the held-out sample."""
    n_books = item_features.shape[0]

    def evaluate():
        model.eval()
        with torch.no_grad():
            item_embs = []
            for i in range(0, n_books, 8192):
                item_embs.append(model.encode_item(item_features[i:i + 8192]))
            all_item_embs = torch.cat(item_embs, dim=0)
            user_embs = model.encode_user(user_features[u_t])
            hits, ndcgs = [], []
            for start in range(0, len(u_t), 64):
                end = min(start + 64, len(u_t))
                scores = user_embs[start:end] @ all_item_embs.T
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


def save_checkpoint(path: Path, model, optimizer, config: dict, hr: float | None = None,
                    ndcg: float | None = None, n_batches: int | None = None):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_hr10": hr,
        "val_ndcg10": ndcg,
        "n_batches_trained": n_batches,
        "config": config,
    }, path)


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    user_features, item_features, feature_set = load_features(args.feature_set, device)
    print(f"Feature set: {feature_set}  user: {tuple(user_features.shape)}  item: {tuple(item_features.shape)}")

    train_dataset = TrainingPairsDataset(
        n_negatives=args.n_negatives,
        data_split="train",
        verbose=False,
        negative_sampling=args.negative_sampling,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset ready: {len(train_dataset):,} positives, {len(train_dataset.exclude):,} users")

    u_arr, b_arr, u_t, b_t = build_eval_sample(args.n_eval_pairs, device)
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
        "feature_set": feature_set,
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

    print(f"Training for up to {args.time_budget}s, eval every {args.eval_every_batches} batches, patience={args.patience}")
    model.train()
    t_start = time.time()
    losses = []
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
            recent = sum(losses[-args.print_every:]) / args.print_every
            print(f"batch {batch_idx:6d}  loss={recent:.4f}  temp={model.log_temperature.exp().item():.2f}  "
                  f"rate={batch_idx / elapsed:.2f} b/s  elapsed={elapsed:.0f}s")

        if batch_idx - last_eval >= args.eval_every_batches:
            hr, ndcg = evaluate()
            print(f"  EVAL batch={batch_idx}  hr@10={hr:.4f}  ndcg@10={ndcg:.4f}  best={best_hr:.4f}")
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

    save_checkpoint(rolling_path, model, optimizer, config, n_batches=batch_idx)
    print(f"\nDone: {batch_idx} batches in {time.time() - t_start:.0f}s, best HR@10={best_hr:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
