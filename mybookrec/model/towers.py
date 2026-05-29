"""Two-tower neural recommendation model.

Each tower is an MLP that maps domain features (user or item) to a shared embedding space.
Similarity between a user and item embedding is the dot product. With L2-normalized outputs,
dot product = cosine similarity, and the model's embeddings are ready for FAISS IndexFlatIP
at inference time without further processing.

The towers don't own the lookup tables. They take raw feature tensors and return embeddings.
The training loop and inference code handle the index-to-feature lookup.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPTower(nn.Module):
    """Shared MLP scaffold for both towers.

    Architecture: input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1], with ReLU + Dropout
    between hidden layers, no activation on the final projection, and L2 normalization on the
    output embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one entry (the output dimension)")

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims[:-1]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, hidden_dims[-1])]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.mlp(features)
        return F.normalize(out, dim=-1)


class ItemTower(_MLPTower):
    """Encoder for book features: [embedding | genre | normalized_pages] -> shared embedding."""


class UserTower(_MLPTower):
    """Encoder for user features: [like_emb | dislike_emb | genre_dist | mean_pages] -> shared embedding."""


class TwoTowerModel(nn.Module):
    """Wraps both towers and computes user-item similarity via dot product.

    The forward signature accepts either:
        item_features shape (B, item_input_dim)      -> similarity shape (B,)
        item_features shape (B, K, item_input_dim)   -> similarity shape (B, K)

    The second shape is what training uses: one positive + K-1 negatives per anchor.
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, hidden_dims, dropout)
        self.item_tower = ItemTower(item_input_dim, hidden_dims, dropout)
        self.embedding_dim = hidden_dims[-1]

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_features)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        return self._similarity(user_emb, item_emb)

    @staticmethod
    def _similarity(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if item_emb.dim() == user_emb.dim() + 1:
            # item shape (B, K, H), user shape (B, H) -> broadcast over K.
            return (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
        return (user_emb * item_emb).sum(dim=-1)
