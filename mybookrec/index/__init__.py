"""FAISS-based nearest-neighbor index over item embeddings."""
from mybookrec.index.faiss_index import build_index, load_index, query, encode_all_items

__all__ = ["build_index", "load_index", "query", "encode_all_items"]
