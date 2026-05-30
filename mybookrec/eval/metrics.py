"""Top-K retrieval metrics for leave-one-out evaluation.

Both metrics take per-user score vectors over the entire item catalog and a single held-out
target item per user. They're batched: pass (B, n_items) scores and (B,) targets, get (B,) results.

Hit Rate @ K: binary — did the target make the top K?
NDCG @ K: rank-weighted — same hit, but with diminishing reward as rank grows.
"""

from __future__ import annotations

import torch


def hit_rate_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Hit Rate @ K — 1 if the target index is in the top-K scoring items, else 0.

    Args:
        scores: shape (B, n_items) — score per item, per user.
        targets: shape (B,) — index of the single relevant item per user.
        k: top-K cutoff.

    Returns:
        Shape (B,) boolean tensor, True where the target was hit.
    """
    top_k_indices = torch.topk(scores, k=k, dim=-1).indices  # (B, K)
    return (top_k_indices == targets.unsqueeze(-1)).any(dim=-1)


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """NDCG @ K with a single relevant target per query (leave-one-out setting).

    For one relevant item at 0-indexed rank r in the top-K list:
        DCG  = 1 / log2(r + 2)
        IDCG = 1 (best case: rank 0 -> 1 / log2(2) = 1)
        NDCG = DCG / IDCG = 1 / log2(r + 2)

    If the target is not in the top K, NDCG = 0.

    Args:
        scores: shape (B, n_items) — score per item, per user.
        targets: shape (B,) — index of the single relevant item per user.
        k: top-K cutoff.

    Returns:
        Shape (B,) float tensor in [0, 1].
    """
    top_k_indices = torch.topk(scores, k=k, dim=-1).indices  # (B, K)
    matches = top_k_indices == targets.unsqueeze(-1)  # (B, K) bool
    found = matches.any(dim=-1)  # (B,) bool
    # argmax returns first-True position when matches has a True; 0 when all False (we mask).
    ranks_zero_indexed = matches.float().argmax(dim=-1)  # (B,) long
    discount = 1.0 / torch.log2(ranks_zero_indexed.float() + 2.0)  # (B,) float
    return discount * found.float()
