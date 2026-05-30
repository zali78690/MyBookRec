"""FAISS index for fast top-K retrieval over item embeddings.

We use IndexFlatIP (exact inner product search):
- L2-normalized item embeddings make inner product == cosine similarity.
- Exact (not approximate) because 1.78M items × 128-dim fits in memory and queries in <10ms.
- Approximate (IndexIVFFlat / IndexHNSW) becomes worthwhile at >100M items.

Usage pattern at inference:
    model = TwoTowerModel(...)
    model.load_state_dict(...)
    item_embs = encode_all_items(model, item_features, device="mps")
    index = build_index(item_embs, save_path="data/checkpoints/book_index.faiss")
    # ...later:
    index = load_index("data/checkpoints/book_index.faiss")
    user_emb = model.encode_user(user_features)
    scores, top_idxs = query(index, user_emb, k=10)
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import torch


def build_index(item_embeddings: np.ndarray, save_path: Path | None = None) -> faiss.Index:
    """Build an IndexFlatIP from L2-normalized item embeddings.

    Args:
        item_embeddings: shape (n_items, dim), float32, unit-norm rows.
        save_path: optional path to serialize the index.

    Returns:
        The FAISS index. Ready for `query()`.
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
    """Load a FAISS index serialized by `build_index`."""
    return faiss.read_index(str(path))


def query(index: faiss.Index, user_embedding: np.ndarray | torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-K nearest neighbors by inner product.

    Args:
        index: FAISS IndexFlatIP.
        user_embedding: shape (dim,) or (n_queries, dim).
        k: number of neighbors per query.

    Returns:
        (scores, indices) — both shape (n_queries, k), float32 / int64.
    """
    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    if user_embedding.dtype != np.float32:
        user_embedding = user_embedding.astype(np.float32)
    if user_embedding.ndim == 1:
        user_embedding = user_embedding[np.newaxis, :]

    return index.search(user_embedding, k)


def encode_all_items(
    model,
    item_features: torch.Tensor,
    batch_size: int = 8192,
) -> np.ndarray:
    """Run every item through the ItemTower and stack the L2-normalized embeddings.

    Args:
        model: a TwoTowerModel (uses model.encode_item).
        item_features: shape (n_items, item_dim) on whatever device the model is on.
        batch_size: how many items per forward pass.

    Returns:
        numpy float32 array shape (n_items, embedding_dim), L2-normalized.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, item_features.shape[0], batch_size):
            chunks.append(model.encode_item(item_features[i:i + batch_size]))
    return torch.cat(chunks, dim=0).cpu().numpy().astype(np.float32)
