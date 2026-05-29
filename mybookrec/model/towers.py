"""Two-tower neural recommendation model.

Each tower is an MLP that maps domain features (user or item) to a shared embedding space.
Similarity between a user and an item embedding is the dot product, scaled by a learnable
temperature so L2-normalized embeddings still yield logits with enough dynamic range for BCE.

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
    output embedding so it lives on the unit sphere of the shared space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
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
    """Wraps both towers and computes user-item similarity.

    L2-normalized dot product is cosine similarity in [-1, 1] — too narrow for BCE to push
    sigmoid toward confident predictions. The learnable temperature `tau` scales it into a
    useful logit range. Stored in log-space so Adam's unconstrained updates can't push it
    negative; the actual temperature is `exp(log_temperature)`.

    Initialized to log(10) so similarity starts at ~10x cosine — already enough range for BCE,
    and the model can grow or shrink it during training.

    Forward signature accepts either:
        item_features shape (B, item_input_dim)      -> similarity shape (B,)
        item_features shape (B, K, item_input_dim)   -> similarity shape (B, K)
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
        init_temperature: float = 10.0,
    ):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, hidden_dims, dropout)
        self.item_tower = ItemTower(item_input_dim, hidden_dims, dropout)
        self.embedding_dim = hidden_dims[-1]
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_features)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        cos = self._cosine(user_emb, item_emb)
        return cos * self.log_temperature.exp()

    @staticmethod
    def _cosine(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if item_emb.dim() == user_emb.dim() + 1:
            return (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
        return (user_emb * item_emb).sum(dim=-1)
