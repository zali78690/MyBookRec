"""Evaluate a trained checkpoint on the held-out test split.

Reports Hit Rate @ K and NDCG @ K (leave-one-out, masking train-rated books).

Usage:
    .venv/bin/python -m mybookrec.eval.evaluate checkpoints/two_tower_v4bce_best.pt
    .venv/bin/python -m mybookrec.eval.evaluate <ckpt> --k 10 --n-pairs 5000 --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mybookrec.eval.metrics import hit_rate_at_k, ndcg_at_k
from mybookrec.io import (
    batch_encode,
    build_train_exclude,
    load_checkpoint,
    load_features_for_checkpoint,
    sample_test_pairs,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace with `checkpoint`, `k`, `n_pairs`, `seed`, `user_batch`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-pairs", type=int, default=5000, help="Held-out test pairs to evaluate on.")
    p.add_argument("--seed", type=int, default=0, help="Fixed seed for reproducible eval samples.")
    p.add_argument("--user-batch", type=int, default=64)
    return p.parse_args()


def main() -> None:
    """Compute HR@K and NDCG@K against the held-out test split."""
    args = parse_args()

    model, config, ckpt = load_checkpoint(args.checkpoint)
    print(
        f"Checkpoint: {args.checkpoint}  trained on {ckpt.get('n_batches_trained', '?'):,} batches  "
        f"saved hr@10={ckpt.get('val_hr10')}"
    )

    device = next(model.parameters()).device.type
    user_features, item_features = load_features_for_checkpoint(config, device)

    t0 = time.time()
    with torch.no_grad():
        all_item_embs = batch_encode(model.encode_item, item_features)
        all_user_embs = batch_encode(model.encode_user, user_features)
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

    hr_mean = hits.float().mean().item()
    print()
    print(f"Hit Rate @ {args.k}:  {hr_mean:.4f}  ({hr_mean * 100:.2f}%)")
    print(f"NDCG @ {args.k}:      {ndcgs.mean().item():.4f}")
    if hits.any():
        print(f"NDCG | hit:         {ndcgs[hits].mean().item():.4f}")


if __name__ == "__main__":
    main()
