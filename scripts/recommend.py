"""End-to-end recommendation pipeline: load checkpoint + FAISS index → top-K with filters.

Pipeline:
  1. Load model + (optionally pre-built) FAISS index.
  2. Compute user embedding from features.
  3. Query FAISS for top-(K * oversample) candidates.
  4. Apply post-ranking filters: min_avg_rating, ebook_only, exclude already-rated.
  5. Return top-K with titles, avg_rating, num_pages, scores.

Usage:
    uv run python scripts/recommend.py checkpoints/two_tower_v4bce_best.pt
    uv run python scripts/recommend.py <ckpt> --top-k 20 --min-avg-rating 4.0 --ebook-only
    uv run python scripts/recommend.py <ckpt> --index checkpoints/two_tower_v4bce_best_index.faiss

If --index isn't provided, the index is built on the fly (~2 sec on MPS).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# macOS-only: FAISS and PyTorch both link against libomp; loading both in the same process
# triggers an OpenMP duplicate-init error. This is the documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import polars as pl
import torch

from mybookrec import DATA_DIR
from mybookrec.index import build_index, encode_all_items, load_index, query
from mybookrec.model.towers import TwoTowerModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--index", type=Path, default=None,
                   help="Pre-built FAISS index (skip on-the-fly encoding).")
    p.add_argument("--user-features", type=Path, default=None,
                   help="Override personal user feature path (defaults to v4 if available).")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--oversample", type=int, default=5,
                   help="Pull K*oversample candidates from FAISS before filtering.")
    p.add_argument("--min-avg-rating", type=float, default=3.5,
                   help="Filter out books below this global average rating.")
    p.add_argument("--ebook-only", action="store_true",
                   help="Restrict to is_ebook=true books only.")
    p.add_argument("--exclude-rated", action="store_true", default=True,
                   help="Exclude books the user has already rated (from my_books.csv).")
    return p.parse_args()


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_user_features_path(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    t = DATA_DIR / "transformed"
    return t / ("user_features_v4.npy" if config["user_input_dim"] == 1163 else "user_features.npy")


def load_item_features(config: dict, device: str) -> torch.Tensor:
    t = DATA_DIR / "transformed"
    if config["item_input_dim"] == 779:
        return torch.from_numpy(np.load(t / "item_features_v4.npy")).to(device).float()
    emb = torch.from_numpy(np.load(t / "book_embeddings.npy")).to(device).float()
    genre = torch.from_numpy(np.load(t / "genre_matrix.npy")).to(device).float()
    pages = torch.from_numpy(np.load(t / "num_pages_normalized.npy")).to(device).float().reshape(-1, 1)
    return torch.cat([emb, genre, pages], dim=1).contiguous()


def get_or_build_index(checkpoint: Path, model: TwoTowerModel, config: dict, device: str,
                       index_path: Path | None):
    """Return (faiss_index, source_description)."""
    if index_path is not None and index_path.exists():
        return load_index(index_path), f"loaded from {index_path.name}"

    item_features = load_item_features(config, device)
    item_embs = encode_all_items(model, item_features)
    return build_index(item_embs), f"built on-the-fly from {item_embs.shape[0]:,} items"


def apply_filters(
    candidate_ids: list[str],
    candidate_scores: list[float],
    meta_dict: dict[str, dict],
    excluded: set[str],
    min_avg_rating: float,
    ebook_only: bool,
) -> list[tuple[str, float]]:
    """Apply post-ranking filters. Returns (book_id, score) pairs that pass."""
    out = []
    for bid, score in zip(candidate_ids, candidate_scores):
        if bid in excluded:
            continue
        row = meta_dict.get(bid, {})
        avg = row.get("average_rating") or 0.0
        if avg < min_avg_rating:
            continue
        if ebook_only and not (row.get("is_ebook") or False):
            continue
        out.append((bid, score))
    return out


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = TwoTowerModel(
        user_input_dim=config["user_input_dim"],
        item_input_dim=config["item_input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # User embedding
    user_features_path = resolve_user_features_path(config, args.user_features)
    user_features = torch.from_numpy(np.load(user_features_path)).to(device).float().unsqueeze(0)
    with torch.no_grad():
        user_emb = model.encode_user(user_features).squeeze(0)
    print(f"User features: {user_features_path.name}  -> {user_emb.shape[0]}-dim embedding")

    # Index
    t0 = time.time()
    index, index_source = get_or_build_index(args.checkpoint, model, config, device, args.index)
    print(f"Index ({index_source}) ready in {time.time() - t0:.1f}s")

    # Query top-(K * oversample) — extra headroom for post-ranking filters
    n_candidates = args.top_k * args.oversample
    scores, indices = query(index, user_emb, k=n_candidates)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    # Look up metadata for the candidates
    t = DATA_DIR / "transformed"
    with open(t / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)
    index_to_book_id = {v: k for k, v in book_id_to_index.items()}
    candidate_ids = [index_to_book_id[i] for i in indices]

    meta = (
        pl.scan_parquet(t / "books_with_genres.parquet")
        .select("book_id", "title", "num_pages", "average_rating", "is_ebook", "genres")
        .filter(pl.col("book_id").is_in(candidate_ids))
        .collect()
    )
    meta_dict = {row["book_id"]: row for row in meta.to_dicts()}

    # Excluded books (already rated)
    excluded: set[str] = set()
    if args.exclude_rated:
        my_books = pl.read_csv(t / "my_books.csv")
        excluded = {str(bid) for bid in my_books["book_id"].to_list()}

    # Filter + take top-K
    filtered = apply_filters(
        candidate_ids, scores, meta_dict, excluded,
        args.min_avg_rating, args.ebook_only,
    )[:args.top_k]

    # Display
    print(f"\nFilters: min_avg_rating={args.min_avg_rating}, ebook_only={args.ebook_only}, "
          f"exclude_rated={args.exclude_rated} ({len(excluded)} books)")
    print(f"Returning top {len(filtered)} of {args.top_k} requested "
          f"({n_candidates} candidates pulled, {n_candidates - len(filtered)} filtered or held back)\n")

    print(f"{'rank':<5} {'score':<8} {'avg':<6} {'pages':<6} {'ebook':<6} {'title'}")
    print("-" * 110)
    for rank, (bid, score) in enumerate(filtered, 1):
        row = meta_dict.get(bid, {})
        title = (row.get("title") or "?")[:75]
        avg = row.get("average_rating", "?")
        pages = row.get("num_pages") or "?"
        is_ebook = row.get("is_ebook")
        ebook_str = "yes" if is_ebook else "no" if is_ebook is False else "?"
        print(f"{rank:<5} {score:<8.4f} {str(avg)[:5]:<6} {str(pages)[:5]:<6} {ebook_str:<6} {title}")


if __name__ == "__main__":
    main()
