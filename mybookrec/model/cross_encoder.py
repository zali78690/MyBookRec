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
    """Joint user+item MLP for top-K re-ranking."""

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ) -> None:
        """Build the joint MLP.

        Args:
            user_input_dim: Width of the user feature vector.
            item_input_dim: Width of the item feature vector.
            hidden_dims: MLP hidden layer widths. Output is always a single logit.
            dropout: Dropout probability between hidden layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = user_input_dim + item_input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Score (user, item) pairs jointly.

        Args:
            user_features: Shape (batch_size, user_dim).
            item_features: Shape (batch_size, item_dim) for one item per user, or
                (batch_size, n_items, item_dim) to score several items per user.

        Returns:
            Raw logit shape (batch_size,) or (batch_size, n_items) matching the items shape.
        """
        if item_features.dim() == 3:
            batch_size, n_items, _ = item_features.shape
            user_repeated = user_features.unsqueeze(1).expand(batch_size, n_items, -1)
            joint = torch.cat([user_repeated, item_features], dim=-1)
            return self.mlp(joint).squeeze(-1)
        joint = torch.cat([user_features, item_features], dim=-1)
        return self.mlp(joint).squeeze(-1)
