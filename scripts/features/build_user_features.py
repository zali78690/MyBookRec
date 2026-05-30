"""Build the personal user feature vector from my_books.csv.

Produces data/transformed/user_features.npy — a single (user_input_dim,) vector
sized to the v1 feature set (779-dim: 384 like + 384 dislike + 10 genre + 1 pages).

Run this after rebuilding item features or after updating your Goodreads CSV.

Usage:
    .venv/bin/python scripts/features/build_user_features.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from mybookrec import DATA_DIR


def load_inputs() -> tuple[pl.DataFrame, dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Load all artifacts the personal-user pipeline depends on.

    Returns:
        Tuple of (my_books csv, book_id→idx map, book_embeddings, genre_matrix, pages_vec).
    """
    transformed = DATA_DIR / "transformed"

    my_books = pl.read_csv(transformed / "my_books.csv")
    with open(transformed / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)

    book_embeddings = np.load(transformed / "book_embeddings.npy")
    genre_matrix = np.load(transformed / "genre_matrix.npy")
    pages_vec = np.load(transformed / "num_pages_normalized.npy")

    print(f"my_books rows:        {len(my_books):,}")
    print(f"book_id_to_index:     {len(book_id_to_index):,} entries")
    print(f"book_embeddings:      {book_embeddings.shape} dtype={book_embeddings.dtype}")
    print(f"genre_matrix:         {genre_matrix.shape}")
    print(f"pages_vec:            {pages_vec.shape}")
    return my_books, book_id_to_index, book_embeddings, genre_matrix, pages_vec


def match_books_to_indices(
    my_books: pl.DataFrame,
    book_id_to_index: dict[str, int],
) -> pl.DataFrame:
    """Map personal book_ids to UCSD catalog row indices, dropping unknowns.

    Args:
        my_books: Personal Goodreads export DataFrame with `book_id` column.
        book_id_to_index: UCSD book_id (string) → row index mapping.

    Returns:
        DataFrame of matched rows with `row_idx` added. Unmatched rows are dropped
        and printed (typically very recent books missing from the UCSD snapshot).
    """
    matched = my_books.with_columns(pl.col("book_id").cast(pl.String).alias("book_id_str")).with_columns(
        pl.col("book_id_str")
        .map_elements(
            lambda b: book_id_to_index.get(b, -1),
            return_dtype=pl.Int64,
        )
        .alias("row_idx")
    )
    dropped = matched.filter(pl.col("row_idx") == -1)
    matched = matched.filter(pl.col("row_idx") != -1)

    print(f"Matched to UCSD:        {len(matched)} of {len(my_books)}")
    print(f"Dropped (not in UCSD):  {len(dropped)}")
    if len(dropped) > 0:
        print(f"\nDropped book_ids: {dropped['book_id'].to_list()}")
    return matched


def build_like_embedding(
    book_embeddings: np.ndarray,
    liked_idx: np.ndarray,
    liked_ratings: np.ndarray,
) -> np.ndarray:
    """Compute the rating-weighted mean of liked-book embeddings.

    Higher ratings pull the centroid toward favorites: 5-stars count more than 4-stars.

    Args:
        book_embeddings: (n_books, dim) full embedding matrix.
        liked_idx: Indices of books rated 4+ stars.
        liked_ratings: Ratings aligned with liked_idx (float32).

    Returns:
        (dim,) float32 like embedding.

    Raises:
        ValueError: If no liked books were matched (cannot compute a centroid).
    """
    if len(liked_idx) == 0:
        raise ValueError("No liked books matched UCSD — cannot build like embedding.")
    liked_embeddings = book_embeddings[liked_idx]
    weights = liked_ratings[:, None]
    weighted = (liked_embeddings * weights).sum(axis=0) / weights.sum()
    return weighted.astype(np.float32)


def build_dislike_embedding(book_embeddings: np.ndarray, disliked_idx: np.ndarray) -> np.ndarray:
    """Compute the simple mean of disliked-book embeddings.

    Returns a zero vector when no disliked books exist — the model treats this as
    "no dislike signal" and contributes zero to similarity for those users.

    Args:
        book_embeddings: (n_books, dim) full embedding matrix.
        disliked_idx: Indices of books rated 1-2 stars.

    Returns:
        (dim,) float32 dislike embedding (zero vector if empty input).
    """
    if len(disliked_idx) == 0:
        print("WARNING: no disliked books matched UCSD. Using zero vector for dislike_emb.")
        return np.zeros(book_embeddings.shape[1], dtype=np.float32)
    return book_embeddings[disliked_idx].mean(axis=0).astype(np.float32)


def build_genre_dist(genre_matrix: np.ndarray, liked_idx: np.ndarray) -> np.ndarray:
    """L2-normalized sum of liked books' genre vectors.

    Args:
        genre_matrix: (n_books, n_genres) per-book L2-normalized vectors.
        liked_idx: Indices of liked books.

    Returns:
        (n_genres,) float32 unit vector (or all zeros if no liked books had genres).
    """
    liked_genres = genre_matrix[liked_idx]
    summed = liked_genres.sum(axis=0)
    norm = np.linalg.norm(summed)
    normalized = summed / norm if norm > 0 else summed
    return normalized.astype(np.float32)


def main() -> None:
    """Build and save the personal user feature vector."""
    my_books, book_id_to_index, book_embeddings, genre_matrix, pages_vec = load_inputs()
    matched = match_books_to_indices(my_books, book_id_to_index)

    liked = matched.filter(pl.col("my_rating") >= 4)
    disliked = matched.filter(pl.col("my_rating") <= 2)
    print(f"Liked (4+):     {len(liked)}")
    print(f"Disliked (1-2): {len(disliked)}")
    print(f"Excluded (3):   {len(matched.filter(pl.col('my_rating') == 3))}")

    liked_idx = liked["row_idx"].to_numpy()
    liked_ratings = liked["my_rating"].to_numpy().astype(np.float32)
    disliked_idx = disliked["row_idx"].to_numpy()

    like_emb = build_like_embedding(book_embeddings, liked_idx, liked_ratings)
    dislike_emb = build_dislike_embedding(book_embeddings, disliked_idx)
    genre_dist = build_genre_dist(genre_matrix, liked_idx)
    mean_pages = float(pages_vec[liked_idx].mean())

    print(f"like_emb    shape={like_emb.shape}  L2={np.linalg.norm(like_emb):.4f}")
    print(f"dislike_emb shape={dislike_emb.shape}  L2={np.linalg.norm(dislike_emb):.4f}")
    print(f"genre_dist  shape={genre_dist.shape}  L2={np.linalg.norm(genre_dist):.4f}")
    print(f"mean_pages: {mean_pages:.4f}")

    # Order matters: this must match what the UserTower expects at training time.
    user_features = np.concatenate(
        [
            like_emb,
            dislike_emb,
            genre_dist,
            np.array([mean_pages], dtype=np.float32),
        ]
    ).astype(np.float32)

    expected_dim = book_embeddings.shape[1] * 2 + genre_matrix.shape[1] + 1
    assert user_features.shape == (expected_dim,), f"Expected ({expected_dim},), got {user_features.shape}"
    assert not np.isnan(user_features).any(), "NaNs in user_features — check upstream"

    output = DATA_DIR / "transformed" / "user_features.npy"
    np.save(output, user_features)
    print(f"Saved user_features {user_features.shape} to {output.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
