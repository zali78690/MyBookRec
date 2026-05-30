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
    ) -> None:
        """Build the MLP layers.

        Args:
            input_dim: Width of the per-row input feature vector.
            hidden_dims: Layer widths from first hidden through final output.
                Last entry is the shared embedding dimension.
            dropout: Dropout probability applied between hidden layers.
        """
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one entry (the output dimension)")

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims[:-1]:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers += [nn.Linear(prev, hidden_dims[-1])]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run features through the MLP and L2-normalize the output.

        Args:
            features: Input tensor of shape (..., input_dim).

        Returns:
            Unit-norm embeddings of shape (..., output_dim).
        """
        return F.normalize(self.mlp(features), dim=-1)


class ItemTower(_MLPTower):
    """Encoder for book features: [embedding | genre | normalized_pages] → shared embedding."""


class UserTower(_MLPTower):
    """Encoder for user features: [like_emb | dislike_emb | genre_dist | mean_pages] → shared embedding."""


class TwoTowerModel(nn.Module):
    """Wraps both towers and computes user-item similarity.

    L2-normalized dot product is cosine similarity in [-1, 1] — too narrow for BCE to push
    sigmoid toward confident predictions. The learnable temperature `tau` scales it into a
    useful logit range. Stored in log-space so Adam's unconstrained updates can't push it
    negative; the actual temperature is `exp(log_temperature)`.

    Initialized to log(10) so similarity starts at ~10x cosine — already enough range for BCE,
    and the model can grow or shrink it during training.

    Forward signature accepts either:
        item_features shape (batch_size, item_input_dim)        → similarity shape (batch_size,)
        item_features shape (batch_size, n_items, item_input_dim) → similarity shape (batch_size, n_items)
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
        init_temperature: float = 10.0,
    ) -> None:
        """Build the two towers and the learnable temperature scalar.

        Args:
            user_input_dim: Width of the user feature vector.
            item_input_dim: Width of the item feature vector.
            hidden_dims: Shared MLP layer widths.
            dropout: Dropout probability applied inside both towers.
            init_temperature: Initial value for `exp(log_temperature)`.
        """
        super().__init__()
        self.user_tower = UserTower(user_input_dim, hidden_dims, dropout)
        self.item_tower = ItemTower(item_input_dim, hidden_dims, dropout)
        self.embedding_dim = hidden_dims[-1]
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features through the UserTower.

        Args:
            user_features: Tensor of shape (..., user_input_dim).

        Returns:
            Unit-norm user embeddings of shape (..., embedding_dim).
        """
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features through the ItemTower.

        Args:
            item_features: Tensor of shape (..., item_input_dim).

        Returns:
            Unit-norm item embeddings of shape (..., embedding_dim).
        """
        return self.item_tower(item_features)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """Compute the temperature-scaled cosine similarity logit.

        Args:
            user_features: (batch_size, user_input_dim).
            item_features: (batch_size, item_input_dim) for one item per user, or
                (batch_size, n_items, item_input_dim) for several items per user.

        Returns:
            Logit shape (batch_size,) or (batch_size, n_items) matching the items shape.
        """
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        cos = self._cosine(user_emb, item_emb)
        return cos * self.log_temperature.exp()

    @staticmethod
    def _cosine(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Dot product of unit-norm tensors with broadcasting over a possible item axis."""
        if item_emb.dim() == user_emb.dim() + 1:
            return (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
        return (user_emb * item_emb).sum(dim=-1)
