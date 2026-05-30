"""Tests for the on-the-fly user feature builder used by the serving path."""

from __future__ import annotations

import numpy as np
import pytest

from mybookrec.serve.schemas import RatingInput
from mybookrec.serve.user_features import compute_user_features_from_ratings


@pytest.fixture
def fixtures() -> dict[str, np.ndarray | dict[str, int]]:
    """A 4-book toy catalog with deterministic embeddings, genres, and pages."""
    return {
        "book_id_to_index": {"a": 0, "b": 1, "c": 2, "d": 3},
        "book_embeddings": np.eye(4, 384, dtype=np.float32),
        "genre_matrix": np.eye(4, 10, dtype=np.float32),
        "num_pages_normalized": np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float32),
    }


def test_expected_use_likes_and_dislikes_populate_user_vector(fixtures) -> None:
    """Expected use: liked + disliked books populate distinct embedding bands."""
    ratings = [
        RatingInput(book_id="a", rating=5),
        RatingInput(book_id="b", rating=4),
        RatingInput(book_id="c", rating=1),
    ]
    vec = compute_user_features_from_ratings(ratings, **fixtures)
    assert vec.shape == (1, 779)
    like_emb = vec[0, :384]
    dislike_emb = vec[0, 384:768]
    genre_dist = vec[0, 768:778]
    pages = vec[0, 778]
    # Liked books a (5★) + b (4★) → rating-weighted mean has non-zero rows 0 and 1.
    assert like_emb[0] > 0
    assert like_emb[1] > 0
    assert like_emb[2] == 0
    # Disliked book c → row 2 non-zero, others zero in dislike band.
    assert dislike_emb[2] > 0
    assert dislike_emb[0] == 0
    # Genre dist is L2-normalised sum of liked rows 0 + 1.
    assert pytest.approx(np.linalg.norm(genre_dist)) == 1.0
    # Mean pages of liked books = (0.1 + 0.2) / 2.
    assert pages == pytest.approx(0.15)


def test_edge_case_unknown_book_ids_are_skipped_silently(fixtures) -> None:
    """Edge case: book_ids not in the catalog don't raise — they're dropped."""
    ratings = [
        RatingInput(book_id="not-in-catalog", rating=5),
        RatingInput(book_id="a", rating=5),
    ]
    vec = compute_user_features_from_ratings(ratings, **fixtures)
    # Only book a contributes — like band has only row 0 set.
    assert vec[0, 0] > 0
    assert vec[0, 1] == 0


def test_failure_case_no_matching_books_returns_zero_vector(fixtures) -> None:
    """Failure case: no book_ids match → all-zero vector (caller should reject)."""
    ratings = [RatingInput(book_id="missing", rating=5)]
    vec = compute_user_features_from_ratings(ratings, **fixtures)
    assert vec.shape == (1, 779)
    assert not np.any(vec)


def test_three_star_ratings_excluded_from_both_bands(fixtures) -> None:
    """Edge case: 3★ counts neither as like nor dislike (matches offline pipeline)."""
    ratings = [RatingInput(book_id="a", rating=3)]
    vec = compute_user_features_from_ratings(ratings, **fixtures)
    assert not np.any(vec)
