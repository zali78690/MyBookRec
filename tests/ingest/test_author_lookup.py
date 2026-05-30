"""Tests for the v4 author-embedding lookup with three-tier fallback."""

from __future__ import annotations

import numpy as np
import pytest

from mybookrec.ingest.author_lookup import (
    AuthorEmbeddingLookup,
    author_key,
    compute_batch_author_means,
)
from mybookrec.ingest.schemas import SilverBook


@pytest.fixture
def lookup() -> AuthorEmbeddingLookup:
    """A 2-author / 3-book trained catalog: A wrote 2 books, B wrote 1."""
    return AuthorEmbeddingLookup(
        isbn13_to_book_id={
            "9780000000001": "book_a1",
            "9780000000002": "book_a2",
            "9780000000003": "book_b1",
        },
        book_id_to_index={"book_a1": 0, "book_a2": 1, "book_b1": 2},
        book_to_author_idx=np.array([0, 0, 1], dtype=np.int64),
        author_embeddings=np.array(
            [
                [1.0, 0.0, 0.0],  # author A
                [0.0, 1.0, 0.0],  # author B
            ],
            dtype=np.float32,
        ),
    )


def make_silver(isbn: str | None, author: str | None) -> SilverBook:
    """Construct a SilverBook with just the fields the lookup cares about."""
    return SilverBook(
        book_id=f"gb_{isbn or author or 'x'}",
        raw_id="x",
        source="google_books",
        title="Some Title",
        authors=[author] if author else [],
        isbn_13=isbn,
    )


def test_isbn_match_returns_trained_embedding(lookup: AuthorEmbeddingLookup) -> None:
    """Expected use: ISBN match → trained author embedding."""
    silver = make_silver(isbn="9780000000001", author="Author A")
    description = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    emb, source = lookup.resolve(silver, description, batch_means={})
    assert source == "trained"
    assert np.allclose(emb, [1.0, 0.0, 0.0])


def test_batch_mean_used_when_no_isbn_but_author_repeats(lookup: AuthorEmbeddingLookup) -> None:
    """Edge case: unknown ISBN, but the author name appears in batch_means → batch mean."""
    silver = make_silver(isbn="9789999999999", author="Unknown Author")
    description = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    batch_means = {"unknown author": np.array([0.2, 0.3, 0.5], dtype=np.float32)}
    emb, source = lookup.resolve(silver, description, batch_means)
    assert source == "batch_mean"
    assert np.allclose(emb, [0.2, 0.3, 0.5])


def test_self_fallback_when_nothing_else_matches(lookup: AuthorEmbeddingLookup) -> None:
    """Failure case: no ISBN match + no batch mean → own description embedding."""
    silver = make_silver(isbn=None, author="Other Author")
    description = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    emb, source = lookup.resolve(silver, description, batch_means={})
    assert source == "self"
    assert np.allclose(emb, [0.1, 0.2, 0.3])


def test_compute_batch_author_means_excludes_singletons() -> None:
    """Edge case: by default, only authors with ≥2 books in the batch contribute a mean."""
    silver_books = [
        make_silver(None, "Author A"),
        make_silver(None, "Author A"),
        make_silver(None, "Author B"),  # singleton — excluded
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    means = compute_batch_author_means(silver_books, embeddings)
    assert "author a" in means
    assert "author b" not in means
    assert np.allclose(means["author a"], [0.5, 0.5, 0.0])


def test_compute_batch_author_means_threshold_is_tunable() -> None:
    """Expected use: lowering min_books_per_author=1 keeps singletons too."""
    books = [make_silver(None, "Solo Author")]
    embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    means = compute_batch_author_means(books, embeddings, min_books_per_author=1)
    assert "solo author" in means


def test_author_key_normalises_case_and_whitespace() -> None:
    """Expected use: author keys are lowercased and trimmed for consistent grouping."""
    assert author_key(make_silver(None, "  Brandon Sanderson ")) == "brandon sanderson"
    assert author_key(make_silver(None, None)) is None


def test_trained_embedding_returns_none_for_unknown_isbn(lookup: AuthorEmbeddingLookup) -> None:
    """Failure case: unknown ISBN → None (no false positive)."""
    assert lookup.trained_embedding_for("9789999999999") is None
    assert lookup.trained_embedding_for(None) is None
