"""Evaluate a trained checkpoint on the held-out test split.

Reports Hit Rate @ K and NDCG @ K (leave-one-out, masking train-rated books).

Usage:
    uv run python scripts/evaluate.py checkpoints/two_tower_v4bce_best.pt
    uv run python scripts/evaluate.py <path> --k 10 --n-pairs 5000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from mybookrec import DATA_DIR
from mybookrec.eval.metrics import hit_rate_at_k, ndcg_at_k
from mybookrec.model.towers import TwoTowerModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-pairs", type=int, default=5000, help="Held-out test pairs to evaluate on.")
    p.add_argument("--seed", type=int, default=0, help="Fixed seed for reproducible eval samples.")
    p.add_argument("--user-batch", type=int, default=64)
    return p.parse_args()


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_features_for_checkpoint(config: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-detect feature set from checkpoint's input dims."""
    t = DATA_DIR / "transformed"
    if config["user_input_dim"] == 1163:
        user_features = torch.from_numpy(np.load(t / "train_user_features_v4.npy")).to(device).float()
        item_features = torch.from_numpy(np.load(t / "item_features_v4.npy")).to(device).float()
    else:
        user_features = torch.from_numpy(np.load(t / "train_user_features.npy")).to(device).float()
        _emb = torch.from_numpy(np.load(t / "book_embeddings.npy")).to(device).float()
        _genre = torch.from_numpy(np.load(t / "genre_matrix.npy")).to(device).float()
        _pages = torch.from_numpy(np.load(t / "num_pages_normalized.npy")).to(device).float().reshape(-1, 1)
        item_features = torch.cat([_emb, _genre, _pages], dim=1).contiguous()
    return user_features, item_features


def build_train_exclude() -> dict[int, np.ndarray]:
    """Per-user set of all train-rated book indices, for masking at eval time."""
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

    train_df = (
        pl.scan_parquet(t / "training_interactions.parquet")
        .filter(pl.col("data_split") == "train")
        .select("user_id", "book_id", "rating")
        .with_columns(pl.col("book_id").cast(pl.String))
        .join(user_map.lazy(), on="user_id", how="left")
        .join(book_map.lazy(), on="book_id", how="left")
        .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
        .collect()
    )
    grouped = train_df.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
    return {row["user_idx"]: np.array(row["rated_books"], dtype=np.int64) for row in grouped.to_dicts()}


def sample_test_pairs(n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample N held-out test (user, positive_book) pairs."""
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
    rng = np.random.default_rng(seed)
    sample = test_df[rng.choice(len(test_df), size=min(n_pairs, len(test_df)), replace=False)]
    return sample["user_idx"].to_numpy(), sample["book_idx"].to_numpy()


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"  trained on {ckpt.get('n_batches_trained', '?'):,} batches  "
          f"feature_set={config.get('feature_set', '?')}  "
          f"saved hr@10={ckpt.get('val_hr10')}")

    model = TwoTowerModel(
        user_input_dim=config["user_input_dim"],
        item_input_dim=config["item_input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    user_features, item_features = load_features_for_checkpoint(config, device)

    t0 = time.time()
    with torch.no_grad():
        item_embs = []
        for i in range(0, item_features.shape[0], 8192):
            item_embs.append(model.encode_item(item_features[i:i + 8192]))
        all_item_embs = torch.cat(item_embs, dim=0)
        user_embs = []
        for i in range(0, user_features.shape[0], 8192):
            user_embs.append(model.encode_user(user_features[i:i + 8192]))
        all_user_embs = torch.cat(user_embs, dim=0)
    print(f"Embeddings precomputed in {time.time() - t0:.1f}s")

    t0 = time.time()
    train_exclude = build_train_exclude()
    print(f"Train-exclude dict for {len(train_exclude):,} users ({time.time() - t0:.1f}s)")

    test_users, test_books = sample_test_pairs(args.n_pairs, args.seed)
    print(f"Sampled {len(test_users):,} test pairs (seed={args.seed})")

    t0 = time.time()
    hits, ndcgs = [], []
    with torch.no_grad():
        for start in range(0, len(test_users), args.user_batch):
            end = min(start + args.user_batch, len(test_users))
            u_batch = test_users[start:end]
            b_batch = test_books[start:end]
            scores = all_user_embs[u_batch] @ all_item_embs.T
            for i, u in enumerate(u_batch):
                excluded = train_exclude.get(int(u))
                if excluded is not None and len(excluded) > 0:
                    scores[i, excluded] = float("-inf")
            targets = torch.from_numpy(b_batch.copy()).to(device)
            hits.append(hit_rate_at_k(scores, targets, k=args.k).cpu())
            ndcgs.append(ndcg_at_k(scores, targets, k=args.k).cpu())
    hits = torch.cat(hits)
    ndcgs = torch.cat(ndcgs)
    print(f"Evaluated {len(hits)} pairs in {time.time() - t0:.0f}s")

    print()
    print(f"Hit Rate @ {args.k}:  {hits.float().mean().item():.4f}  ({hits.float().mean().item() * 100:.2f}%)")
    print(f"NDCG @ {args.k}:      {ndcgs.mean().item():.4f}")
    if hits.any():
        print(f"NDCG | hit:         {ndcgs[hits].mean().item():.4f}")


if __name__ == "__main__":
    main()
