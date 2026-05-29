"""Cross-encoder for two-stage retrieval-then-ranking.

The two-tower model is a *retrieval* model: it scores user-item pairs via dot product
of independently-computed embeddings. Fast at inference (precompute item embeddings
once, score via single matmul) but unable to model fine-grained user-item interactions.

A cross-encoder takes (user_features, item_features) jointly — concatenates them and
runs an MLP — so every layer can model user-item interactions. This is too expensive
to score against all 1.78M items per user. Used as a re-ranker on a small candidate
pool (e.g., top-100 from the two-tower retrieval) to lift precision-at-top.

Standard production pattern: retrieval (two-tower) → narrow 1.78M to ~100 → re-rank
(cross-encoder) → final top-10.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossEncoder(nn.Module):
    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = user_input_dim + item_input_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Score (user, item) pairs jointly.

        Args:
            user_features: shape (B, user_dim).
            item_features: shape (B, item_dim) or (B, K, item_dim).
                The latter is the broadcast case — score each user against K items.

        Returns:
            (B,) or (B, K) score tensor (raw logits, not probabilities).
        """
        if item_features.dim() == 3:
            B, K, _ = item_features.shape
            user_repeated = user_features.unsqueeze(1).expand(B, K, -1)
            x = torch.cat([user_repeated, item_features], dim=-1)
            return self.mlp(x).squeeze(-1)
        x = torch.cat([user_features, item_features], dim=-1)
        return self.mlp(x).squeeze(-1)
