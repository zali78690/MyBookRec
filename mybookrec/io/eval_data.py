"""Eval-only IO: training-interactions reader, exclude dict, test-pair sampler.

Used by training (to mask train-rated books when computing held-out HR@K) and
by the evaluate CLI. Not loaded by serving or inference — those paths don't need
the training-split machinery.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mybookrec import DATA_DIR
from mybookrec.io.artifacts import TransformedArtifacts


def id_map_df(mapping: dict[str, int], id_col: str, idx_col: str) -> pl.DataFrame:
    """Materialise an id-to-index dict as a typed Polars DataFrame for join use.

    Args:
        mapping: Dict of {string id: integer index}.
        id_col: Name of the id column in the output frame.
        idx_col: Name of the index column in the output frame.

    Returns:
        Polars DataFrame with one row per mapping entry.
    """
    return pl.DataFrame(
        {id_col: list(mapping.keys()), idx_col: list(mapping.values())},
        schema={id_col: pl.String, idx_col: pl.Int64},
    )


def load_split_interactions(
    data_split: str,
    artifacts: TransformedArtifacts | None = None,
    min_rating: int | None = None,
) -> pl.DataFrame:
    """Load a slice of training_interactions joined onto integer indices.

    Args:
        data_split: One of "train", "validation", "test".
        artifacts: Source of id mappings. Defaults to one bound at DATA_DIR / "transformed".
        min_rating: If set, keep only rows with rating >= this value.

    Returns:
        Polars DataFrame with columns user_id, book_id, rating, user_idx, book_idx.
    """
    if artifacts is None:
        artifacts = TransformedArtifacts(DATA_DIR / "transformed")

    user_map = id_map_df(artifacts.user_id_to_index, "user_id", "user_idx")
    book_map = id_map_df(artifacts.book_id_to_index, "book_id", "book_idx")

    q = (
        pl.scan_parquet(DATA_DIR / "transformed" / "training_interactions.parquet")
        .filter(pl.col("data_split") == data_split)
        .select("user_id", "book_id", "rating")
        .with_columns(pl.col("book_id").cast(pl.String))
        .join(user_map.lazy(), on="user_id", how="left")
        .join(book_map.lazy(), on="book_id", how="left")
        .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
    )
    if min_rating is not None:
        q = q.filter(pl.col("rating") >= min_rating)
    return q.collect()


def build_train_exclude() -> dict[int, np.ndarray]:
    """Map each user to the set of book indices they rated in the train split.

    Used at eval time to mask already-rated books from candidate recommendations.

    Returns:
        Dict from user_idx to a numpy int64 array of book indices.
    """
    train_df = load_split_interactions("train")
    grouped = train_df.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
    return {row["user_idx"]: np.array(row["rated_books"], dtype=np.int64) for row in grouped.to_dicts()}


def sample_test_pairs(n_pairs: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample held-out test (user_idx, positive_book_idx) pairs.

    Args:
        n_pairs: How many pairs to sample (or all if fewer exist).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (user_idxs, book_idxs) numpy arrays.
    """
    test_df = load_split_interactions("test", min_rating=4)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_df), size=min(n_pairs, len(test_df)), replace=False)
    sample = test_df[indices]
    return sample["user_idx"].to_numpy(), sample["book_idx"].to_numpy()
