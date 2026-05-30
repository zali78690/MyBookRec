"""End-to-end recommendation pipeline: checkpoint + FAISS index → top-K with filters.

Pipeline:
    1. Load model + (optionally pre-built) FAISS index.
    2. Compute user embedding from the personal user features.
    3. Query FAISS for top-(K * oversample) candidates.
    4. Apply post-ranking filters: min_avg_rating, ebook_only, exclude already-rated.
    5. Print top-K with titles, scores, and metadata.

Usage:
    .venv/bin/python scripts/recommend.py checkpoints/two_tower_v4bce_best.pt
    .venv/bin/python scripts/recommend.py <ckpt> --top-k 20 --min-avg-rating 4.0 --ebook-only
    .venv/bin/python scripts/recommend.py <ckpt> --index checkpoints/two_tower_v4bce_best_index.faiss

If --index isn't provided, the index is built on the fly (~2s on MPS).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# macOS libomp conflict between FAISS and PyTorch — documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import polars as pl
import torch

from mybookrec import DATA_DIR
from mybookrec.index import build_index, encode_all_items, load_index, query
from mybookrec.io import load_checkpoint, load_item_features, personal_user_features_path
from mybookrec.model.towers import TwoTowerModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace with checkpoint, index, top_k, oversample, filter flags.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--index", type=Path, default=None, help="Pre-built FAISS index (skip on-the-fly encoding).")
    p.add_argument(
        "--user-features",
        type=Path,
        default=None,
        help="Override personal user feature path (defaults to v4 if available).",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--oversample", type=int, default=5, help="Pull K*oversample candidates from FAISS before filtering."
    )
    p.add_argument(
        "--min-avg-rating", type=float, default=3.5, help="Filter out books below this global average rating."
    )
    p.add_argument("--ebook-only", action="store_true", help="Restrict to is_ebook=true books only.")
    p.add_argument(
        "--exclude-rated",
        action="store_true",
        default=True,
        help="Exclude books the user has already rated (from my_books.csv).",
    )
    return p.parse_args()


def get_or_build_index(
    model: TwoTowerModel,
    config: dict,
    device: str,
    index_path: Path | None,
) -> tuple[faiss.Index, str]:
    """Load the FAISS index from disk if available, otherwise build it from the model.

    Args:
        model: A TwoTowerModel for on-the-fly item encoding.
        config: Checkpoint config dict (used to load matching item features).
        device: Target torch device.
        index_path: If provided and exists, load instead of building.

    Returns:
        Tuple of (index, source_description_for_logging).
    """
    if index_path is not None and index_path.exists():
        return load_index(index_path), f"loaded from {index_path.name}"
    item_features = load_item_features(config, device)
    item_embeddings = encode_all_items(model, item_features)
    return build_index(item_embeddings), f"built on-the-fly from {item_embeddings.shape[0]:,} items"


def apply_filters(
    candidate_ids: list[str],
    candidate_scores: list[float],
    meta_dict: dict[str, dict],
    excluded: set[str],
    min_avg_rating: float,
    ebook_only: bool,
) -> list[tuple[str, float]]:
    """Apply post-ranking filters to FAISS candidates.

    Args:
        candidate_ids: Book ids ranked by similarity (highest first).
        candidate_scores: Similarity scores aligned with candidate_ids.
        meta_dict: Book id → metadata row (must contain average_rating, is_ebook).
        excluded: Book ids to drop unconditionally (e.g. already-rated).
        min_avg_rating: Drop books below this average rating.
        ebook_only: If True, drop non-ebook results.

    Returns:
        Filtered (book_id, score) pairs, preserving original rank order.
    """
    out: list[tuple[str, float]] = []
    for bid, score in zip(candidate_ids, candidate_scores):
        if bid in excluded:
            continue
        row = meta_dict.get(bid, {})
        if (row.get("average_rating") or 0.0) < min_avg_rating:
            continue
        if ebook_only and not (row.get("is_ebook") or False):
            continue
        out.append((bid, score))
    return out


def format_ebook(value: object) -> str:
    """Render the is_ebook tri-state ('?' for missing) as a short string.

    Args:
        value: Cell value from books_with_genres (True, False, or None).

    Returns:
        "yes", "no", or "?".
    """
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "?"


def main() -> None:
    """Run the retrieve-then-filter pipeline and print top-K recommendations."""
    args = parse_args()

    model, config, _ = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device.type
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    user_features_path = args.user_features or personal_user_features_path(config)
    user_features = torch.from_numpy(np.load(user_features_path)).to(device).float().unsqueeze(0)
    with torch.no_grad():
        user_emb = model.encode_user(user_features).squeeze(0)
    print(f"User features: {user_features_path.name}  -> {user_emb.shape[0]}-dim embedding")

    t0 = time.time()
    index, index_source = get_or_build_index(model, config, device, args.index)
    print(f"Index ({index_source}) ready in {time.time() - t0:.1f}s")

    n_candidates = args.top_k * args.oversample
    scores_arr, indices_arr = query(index, user_emb, k=n_candidates)
    scores = scores_arr[0].tolist()
    indices = indices_arr[0].tolist()

    transformed = DATA_DIR / "transformed"
    with open(transformed / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)
    index_to_book_id = {v: k for k, v in book_id_to_index.items()}
    candidate_ids = [index_to_book_id[i] for i in indices]

    meta = (
        pl.scan_parquet(transformed / "books_with_genres.parquet")
        .select("book_id", "title", "num_pages", "average_rating", "is_ebook", "genres")
        .filter(pl.col("book_id").is_in(candidate_ids))
        .collect()
    )
    meta_dict = {row["book_id"]: row for row in meta.to_dicts()}

    excluded: set[str] = set()
    if args.exclude_rated:
        my_books = pl.read_csv(transformed / "my_books.csv")
        excluded = {str(bid) for bid in my_books["book_id"].to_list()}

    filtered = apply_filters(
        candidate_ids,
        scores,
        meta_dict,
        excluded,
        args.min_avg_rating,
        args.ebook_only,
    )[: args.top_k]

    print(
        f"\nFilters: min_avg_rating={args.min_avg_rating}, ebook_only={args.ebook_only}, "
        f"exclude_rated={args.exclude_rated} ({len(excluded)} books)"
    )
    print(
        f"Returning top {len(filtered)} of {args.top_k} requested "
        f"({n_candidates} candidates pulled, {n_candidates - len(filtered)} filtered or held back)\n"
    )

    print(f"{'rank':<5} {'score':<8} {'avg':<6} {'pages':<6} {'ebook':<6} {'title'}")
    print("-" * 110)
    for rank, (bid, score) in enumerate(filtered, 1):
        row = meta_dict.get(bid, {})
        title = (row.get("title") or "?")[:75]
        avg = row.get("average_rating", "?")
        pages = row.get("num_pages") or "?"
        ebook_str = format_ebook(row.get("is_ebook"))
        print(f"{rank:<5} {score:<8.4f} {str(avg)[:5]:<6} {str(pages)[:5]:<6} {ebook_str:<6} {title}")


if __name__ == "__main__":
    main()
