"""FAISS index for fast top-K item retrieval.

IndexFlatIP gives exact inner-product search over L2-normalized item embeddings,
which is cosine similarity. Exact (not approximate) is fine at our 1.78M item scale —
queries take <10ms. IndexIVFFlat / IndexHNSW becomes worthwhile beyond ~100M items.

Typical usage::

    from mybookrec.io import load_checkpoint, load_item_features
    from mybookrec.index import build_index, encode_all_items, query

    model, config, _ = load_checkpoint("checkpoints/two_tower_v4bce_best.pt")
    item_features = load_item_features(config, device="mps")
    item_embeddings = encode_all_items(model, item_features)
    index = build_index(item_embeddings, save_path="checkpoints/book_index.faiss")

    user_emb = model.encode_user(user_features)
    scores, top_idxs = query(index, user_emb, k=10)
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch

from mybookrec.io import batch_encode
from mybookrec.model.towers import TwoTowerModel


def build_index(item_embeddings: np.ndarray, save_path: Path | None = None) -> faiss.Index:
    """Build an IndexFlatIP from L2-normalized item embeddings.

    Args:
        item_embeddings: Shape (n_items, embedding_dim). Cast to float32 if needed.
            Rows should already be L2-normalized so that inner product == cosine.
        save_path: Optional path to serialize the index for reuse.

    Returns:
        FAISS IndexFlatIP, ready for `query`.
    """
    if item_embeddings.dtype != np.float32:
        item_embeddings = item_embeddings.astype(np.float32)

    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))

    return index


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index serialized by `build_index`.

    Args:
        path: Path to the .faiss file.

    Returns:
        The FAISS index.
    """
    return faiss.read_index(str(path))


def query(
    index: faiss.Index,
    user_embedding: np.ndarray | torch.Tensor,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K nearest neighbors by inner product.

    Args:
        index: A FAISS IndexFlatIP populated with item embeddings.
        user_embedding: Shape (embedding_dim,) for a single query or (n, embedding_dim) for batched.
        k: Number of neighbors per query.

    Returns:
        Tuple of (scores, indices), both shape (n_queries, k).
    """
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    if user_embedding.dtype != np.float32:
        user_embedding = user_embedding.astype(np.float32)
    if user_embedding.ndim == 1:
        user_embedding = user_embedding[np.newaxis, :]
    return index.search(user_embedding, k)


def encode_all_items(
    model: TwoTowerModel,
    item_features: torch.Tensor,
    batch_size: int = 8192,
) -> np.ndarray:
    """Encode every item through the trained ItemTower into a numpy array.

    Wraps the shared `batch_encode` to add device-detachment so the result is a
    numpy array ready to feed into `build_index`.

    Args:
        model: A TwoTowerModel (uses `model.encode_item`).
        item_features: Item feature matrix on the same device as `model`.
        batch_size: Rows per forward pass.

    Returns:
        Numpy float32 array of shape (n_items, embedding_dim), L2-normalized.
    """
    model.eval()
    with torch.no_grad():
        embeddings = batch_encode(model.encode_item, item_features, batch_size=batch_size)
    return embeddings.cpu().numpy().astype(np.float32)
