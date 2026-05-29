"""Loss functions for two-tower training.

InfoNCE (info noise contrastive estimation) is the standard rec-sys loss: a categorical
cross-entropy where each anchor's positive item competes against many negatives. With
L2-normalized embeddings and a temperature scalar, this is equivalent to the CLIP loss
and the sampled-softmax used in the YouTube DNN paper.

For two-tower retrieval, in-batch negatives are free: every other user's positive item
in the batch is a negative for you. With batch_size=B, each anchor sees B-1 in-batch
negatives + N sampled negatives.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_in_batch(
    user_embs: torch.Tensor,
    pos_item_embs: torch.Tensor,
    neg_item_embs: torch.Tensor,
    temperature: torch.Tensor | float,
    pos_item_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE loss with in-batch negatives + explicit sampled negatives.

    Each anchor competes against:
      - its own positive item (1)
      - every other anchor's positive item in the batch (B - 1) — in-batch negatives
      - N explicitly-sampled negative items per anchor

    The target is "match your own positive" — implemented as cross-entropy where the
    target index for anchor i is i (the diagonal of the in-batch similarity matrix).

    Critical collision handling: if two anchors in the batch share the same positive item
    (e.g. both users love Harry Potter), the off-diagonal entry would treat user_j's
    positive as user_i's negative, penalizing a correct match. With B=1024 and
    long-tail popularity, this happens ~hundreds of times per batch. We mask those
    off-diagonal collisions to -inf via `pos_item_idx` so they're excluded from softmax.

    Args:
        user_embs: shape (B, H), unit-norm user embeddings.
        pos_item_embs: shape (B, H), unit-norm positive item embeddings.
        neg_item_embs: shape (B, N, H), unit-norm sampled negative item embeddings.
        temperature: scalar multiplier on similarities (higher = sharper distribution).
        pos_item_idx: shape (B,) int tensor of positive item indices for collision masking.
            If None, collisions are not masked (faster but slightly biased).

    Returns:
        Scalar loss.
    """
    B = user_embs.shape[0]
    in_batch_sim = user_embs @ pos_item_embs.T  # (B, B)
    sampled_sim = torch.einsum("bh,bnh->bn", user_embs, neg_item_embs)  # (B, N)
    # CRITICAL: scale by temperature BEFORE masking. Otherwise -inf entries get multiplied
    # by temperature, and the gradient of `-inf * temperature` w.r.t. temperature is NaN
    # (which then propagates to every parameter via Adam's update). Masking the scaled logits
    # instead gives a clean gradient — masked entries are -inf in forward, contribute zero to
    # softmax, and gradient at those positions is exactly zero (no temperature dependency).
    logits = torch.cat([in_batch_sim, sampled_sim], dim=1) * temperature  # (B, B + N)

    if pos_item_idx is not None:
        same_item = pos_item_idx.unsqueeze(0) == pos_item_idx.unsqueeze(1)  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        collision_mask = same_item & ~eye  # off-diagonal collisions only
        pad = torch.zeros(B, sampled_sim.shape[1], dtype=torch.bool, device=logits.device)
        full_mask = torch.cat([collision_mask, pad], dim=1)  # (B, B + N)
        logits = logits.masked_fill(full_mask, float("-inf"))

    targets = torch.arange(B, device=user_embs.device)
    return F.cross_entropy(logits, targets)
