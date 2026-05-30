"""On-the-fly user feature construction for the real-time serving path.

Mirrors `mybookrec.features.build_user_features` but operates on a list of
in-flight ratings instead of reading my_books.csv. Produces a (1, 779) float32
vector compatible with the v1 feature set.
"""

from __future__ import annotations

import numpy as np

from mybookrec.serve.schemas import RatingInput

LIKE_MIN_RATING = 4
DISLIKE_MAX_RATING = 2


def compute_user_features_from_ratings(
    ratings: list[RatingInput],
    book_id_to_index: dict[str, int],
    book_embeddings: np.ndarray,
    genre_matrix: np.ndarray,
    num_pages_normalized: np.ndarray,
) -> np.ndarray:
    """Build a single (1, 779) user feature vector for the v1 feature set.

    Concatenation order matches the training pipeline:
    `[like_emb (384) | dislike_emb (384) | genre_dist (10) | mean_pages (1)]`.

    Args:
        ratings: User's rated books. Entries whose `book_id` is missing from
            `book_id_to_index` are skipped silently.
        book_id_to_index: UCSD book_id (string) → row index mapping.
        book_embeddings: Full (n_books, 384) embedding matrix.
        genre_matrix: Full (n_books, 10) L2-normalized genre matrix.
        num_pages_normalized: 1-D (n_books,) normalized page counts.

    Returns:
        Float32 numpy array of shape (1, 779) ready to feed `model.encode_user`.
    """
    liked_idx: list[int] = []
    liked_ratings: list[float] = []
    disliked_idx: list[int] = []
    for r in ratings:
        idx = book_id_to_index.get(r.book_id)
        if idx is None:
            continue
        if r.rating >= LIKE_MIN_RATING:
            liked_idx.append(idx)
            liked_ratings.append(float(r.rating))
        elif r.rating <= DISLIKE_MAX_RATING:
            disliked_idx.append(idx)

    dim = book_embeddings.shape[1]
    n_genres = genre_matrix.shape[1]

    if liked_idx:
        liked_arr = np.asarray(liked_idx, dtype=np.int64)
        weights = np.asarray(liked_ratings, dtype=np.float32)[:, None]
        like_emb = (book_embeddings[liked_arr] * weights).sum(axis=0) / weights.sum()
        summed_genres = genre_matrix[liked_arr].sum(axis=0)
        norm = np.linalg.norm(summed_genres)
        genre_dist = (summed_genres / norm) if norm > 0 else summed_genres
        mean_pages = float(num_pages_normalized[liked_arr].mean())
    else:
        like_emb = np.zeros(dim, dtype=np.float32)
        genre_dist = np.zeros(n_genres, dtype=np.float32)
        mean_pages = 0.0

    if disliked_idx:
        disliked_arr = np.asarray(disliked_idx, dtype=np.int64)
        dislike_emb = book_embeddings[disliked_arr].mean(axis=0)
    else:
        dislike_emb = np.zeros(dim, dtype=np.float32)

    vec = np.concatenate(
        [
            like_emb.astype(np.float32),
            dislike_emb.astype(np.float32),
            genre_dist.astype(np.float32),
            np.array([mean_pages], dtype=np.float32),
        ]
    )
    return vec.reshape(1, -1).astype(np.float32)
