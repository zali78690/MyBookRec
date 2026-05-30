"""Build the bulk training-user feature matrix from train-split interactions.

Sparse matmul over the user×book interaction grid (99.99% empty) gives each
user's like/dislike/genre/pages aggregates simultaneously. Per-user Python loops
would take ~10 minutes; sparse @ dense matmul finishes in seconds.

Writes:
- data/transformed/train_user_features.npy — (n_users, 779) for v1 feature set.
- data/transformed/user_id_to_index.json — compact user_id → row mapping for the
  filtered set (users with at least one liked book).

Usage:
    .venv/bin/python scripts/features/build_train_user_features.py
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from mybookrec import DATA_DIR


def load_item_side() -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Load item-side artifacts produced by build_item_features + embed.ipynb.

    Returns:
        Tuple of (book_id_to_index, book_embeddings, genre_matrix, pages_vec).
    """
    transformed = DATA_DIR / "transformed"
    with open(transformed / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)

    book_embeddings = np.load(transformed / "book_embeddings.npy").astype(np.float32)
    genre_matrix = np.load(transformed / "genre_matrix.npy").astype(np.float32)
    pages_vec = np.load(transformed / "num_pages_normalized.npy").astype(np.float32)
    print(
        f"Books: {book_embeddings.shape[0]:,} | embedding_dim: {book_embeddings.shape[1]} | "
        f"n_genres: {genre_matrix.shape[1]}"
    )
    return book_id_to_index, book_embeddings, genre_matrix, pages_vec


def load_train_interactions(book_id_to_index: dict[str, int]) -> tuple[pl.DataFrame, list[str]]:
    """Load train-split interactions and assign integer indices to users + books.

    Args:
        book_id_to_index: Pre-built book mapping (excludes catalog filtering).

    Returns:
        Tuple of (interactions DataFrame with user_idx/book_idx columns, list of
        unique user_id hashes ordered by their assigned indices).
    """
    transformed = DATA_DIR / "transformed"
    interactions = (
        pl.scan_parquet(transformed / "books_with_interactions.parquet")
        .filter(pl.col("data_split") == "train")
        .select("user_id", "book_id", "rating")
        .collect()
    )
    print(f"Train interactions: {len(interactions):,}")

    interactions = (
        interactions.with_columns(pl.col("book_id").cast(pl.String))
        .with_columns(
            pl.col("book_id")
            .map_elements(
                lambda b: book_id_to_index.get(b, -1),
                return_dtype=pl.Int64,
            )
            .alias("book_idx")
        )
        .filter(pl.col("book_idx") >= 0)
    )

    unique_users = interactions["user_id"].unique().to_list()
    user_id_to_index = {uid: i for i, uid in enumerate(unique_users)}
    interactions = interactions.with_columns(
        pl.col("user_id")
        .map_elements(
            lambda u: user_id_to_index[u],
            return_dtype=pl.Int64,
        )
        .alias("user_idx")
    )
    print(f"Unique train users: {len(unique_users):,}  Interactions after book lookup: {len(interactions):,}")
    return interactions, unique_users


def build_interaction_matrices(
    interactions: pl.DataFrame,
    n_users: int,
    n_books: int,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build three sparse (n_users, n_books) matrices for like/dislike aggregations.

    Args:
        interactions: DataFrame with user_idx, book_idx, rating columns.
        n_users: Number of unique users.
        n_books: Number of books in the catalog.

    Returns:
        Tuple of:
            w_like — rating-weighted (rating where rating >= 4).
            m_like — binary mask of liked books (rating >= 4).
            m_dislike — binary mask of disliked books (rating <= 2).
    """
    user_idx = interactions["user_idx"].to_numpy()
    book_idx = interactions["book_idx"].to_numpy()
    rating = interactions["rating"].to_numpy().astype(np.float32)
    liked = rating >= 4
    disliked = rating <= 2

    w_like = csr_matrix(
        (rating[liked], (user_idx[liked], book_idx[liked])),
        shape=(n_users, n_books),
    )
    m_like = csr_matrix(
        (np.ones(liked.sum(), dtype=np.float32), (user_idx[liked], book_idx[liked])),
        shape=(n_users, n_books),
    )
    m_dislike = csr_matrix(
        (np.ones(disliked.sum(), dtype=np.float32), (user_idx[disliked], book_idx[disliked])),
        shape=(n_users, n_books),
    )
    print(f"w_like nnz: {w_like.nnz:,}  m_like nnz: {m_like.nnz:,}  m_dislike nnz: {m_dislike.nnz:,}")
    return w_like, m_like, m_dislike


def compute_user_features(
    w_like: csr_matrix,
    m_like: csr_matrix,
    m_dislike: csr_matrix,
    book_embeddings: np.ndarray,
    genre_matrix: np.ndarray,
    pages_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the four user feature channels via sparse matmul.

    Args:
        w_like: Rating-weighted like sparse matrix.
        m_like: Binary like sparse matrix.
        m_dislike: Binary dislike sparse matrix.
        book_embeddings: (n_books, embedding_dim) item embeddings.
        genre_matrix: (n_books, n_genres) per-book L2-normed vectors.
        pages_vec: (n_books,) normalized page values.

    Returns:
        Tuple of (like_emb, dislike_emb, genre_dist, mean_pages, liked_count).
        liked_count is returned for the downstream valid-user filter.
    """
    like_weight_sum = np.asarray(w_like.sum(axis=1)).flatten()
    dislike_count = np.asarray(m_dislike.sum(axis=1)).flatten()
    liked_count = np.asarray(m_like.sum(axis=1)).flatten()

    safe_like_weight = np.where(like_weight_sum > 0, like_weight_sum, 1.0).astype(np.float32)
    safe_dislike_count = np.where(dislike_count > 0, dislike_count, 1.0).astype(np.float32)
    safe_liked_count = np.where(liked_count > 0, liked_count, 1.0).astype(np.float32)

    like_emb = (w_like @ book_embeddings) / safe_like_weight[:, None]
    dislike_emb = (m_dislike @ book_embeddings) / safe_dislike_count[:, None]

    genre_unnorm = m_like @ genre_matrix
    genre_norms = np.linalg.norm(genre_unnorm, axis=1, keepdims=True)
    safe_genre_norms = np.where(genre_norms > 0, genre_norms, 1.0).astype(np.float32)
    genre_dist = genre_unnorm / safe_genre_norms

    mean_pages = ((m_like @ pages_vec.reshape(-1, 1)).flatten() / safe_liked_count).astype(np.float32)

    print(f"Users with no disliked books: {int((dislike_count == 0).sum()):,}")
    print(f"Users with no liked books:    {int((liked_count == 0).sum()):,}  (will be dropped)")
    return like_emb, dislike_emb, genre_dist, mean_pages, liked_count


def assemble_features(
    like_emb: np.ndarray,
    dislike_emb: np.ndarray,
    genre_dist: np.ndarray,
    mean_pages: np.ndarray,
    liked_count: np.ndarray,
    embedding_dim: int,
    n_genres: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate channels and filter to users with at least one liked book.

    Args:
        like_emb: (n_users, embedding_dim) like channel.
        dislike_emb: (n_users, embedding_dim) dislike channel.
        genre_dist: (n_users, n_genres) genre channel.
        mean_pages: (n_users,) pages channel.
        liked_count: (n_users,) used as the "valid user" filter.
        embedding_dim: Embedding dimensionality (for shape assertion).
        n_genres: Number of genres (for shape assertion).

    Returns:
        Tuple of (user_features matrix shape (n_valid, expected_dim),
        valid_old_indices array mapping new row → original user_idx).
    """
    valid_mask = liked_count > 0
    user_features = np.concatenate(
        [
            like_emb[valid_mask],
            dislike_emb[valid_mask],
            genre_dist[valid_mask],
            mean_pages[valid_mask].reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    expected_dim = embedding_dim * 2 + n_genres + 1
    n_valid = int(valid_mask.sum())
    assert user_features.shape == (n_valid, expected_dim), (
        f"Expected ({n_valid}, {expected_dim}), got {user_features.shape}"
    )
    assert not np.isnan(user_features).any(), "NaNs in user_features — check guard divisions"

    return user_features, np.where(valid_mask)[0]


def print_sanity_checks(user_features: np.ndarray, embedding_dim: int, n_genres: int) -> None:
    """Print per-channel norm/distribution summaries to catch upstream bugs.

    Args:
        user_features: Final concatenated user feature matrix.
        embedding_dim: Width of each embedding channel.
        n_genres: Width of the genre channel.
    """
    like_norms = np.linalg.norm(user_features[:, :embedding_dim], axis=1)
    dislike_norms = np.linalg.norm(user_features[:, embedding_dim : 2 * embedding_dim], axis=1)
    genre_norms = np.linalg.norm(user_features[:, 2 * embedding_dim : 2 * embedding_dim + n_genres], axis=1)
    pages_col = user_features[:, -1]
    print("\nSanity checks:")
    print(f"  like_emb    norm: mean={like_norms.mean():.3f}  std={like_norms.std():.3f}")
    print(f"  dislike_emb norm: mean={dislike_norms.mean():.3f}  zero={(dislike_norms == 0).sum():,}")
    print(f"  genre_dist  norm: mean={genre_norms.mean():.3f}  (≈1.0 expected)")
    print(f"  mean_pages: mean={pages_col.mean():.3f}  range=[{pages_col.min():.3f}, {pages_col.max():.3f}]")


def main() -> None:
    """Build the bulk train-user features matrix and save with the compact id mapping."""
    book_id_to_index, book_embeddings, genre_matrix, pages_vec = load_item_side()
    interactions, unique_users = load_train_interactions(book_id_to_index)

    n_users = len(unique_users)
    n_books = book_embeddings.shape[0]
    embedding_dim = book_embeddings.shape[1]
    n_genres = genre_matrix.shape[1]

    w_like, m_like, m_dislike = build_interaction_matrices(interactions, n_users, n_books)
    like_emb, dislike_emb, genre_dist, mean_pages, liked_count = compute_user_features(
        w_like,
        m_like,
        m_dislike,
        book_embeddings,
        genre_matrix,
        pages_vec,
    )
    user_features, valid_old_indices = assemble_features(
        like_emb,
        dislike_emb,
        genre_dist,
        mean_pages,
        liked_count,
        embedding_dim,
        n_genres,
    )

    compact_id_map = {unique_users[old_idx]: new_idx for new_idx, old_idx in enumerate(valid_old_indices)}

    transformed = DATA_DIR / "transformed"
    np.save(transformed / "train_user_features.npy", user_features)
    with open(transformed / "user_id_to_index.json", "w") as f:
        json.dump(compact_id_map, f)

    print(f"\nSaved train_user_features {user_features.shape}")
    print(f"Saved user_id_to_index with {len(compact_id_map):,} entries")
    print(f"Dropped users with no liked books: {n_users - len(compact_id_map):,}")
    print_sanity_checks(user_features, embedding_dim, n_genres)


if __name__ == "__main__":
    main()
