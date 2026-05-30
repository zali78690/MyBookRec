"""Tests for the bronze→silver pipeline (dedup + parquet write)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from mybookrec.ingest import to_silver
from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings


def make_book(book_id: str, source: str, *, isbn_13: str | None = None, ratings_count: int = 0) -> SilverBook:
    """Construct a minimal SilverBook for dedup tests."""
    return SilverBook(
        book_id=book_id,
        raw_id=book_id.split("_", 1)[-1],
        source=source,
        title="The Final Empire",
        authors=["Brandon Sanderson"],
        isbn_13=isbn_13,
        ratings_count=ratings_count,
    )


def test_dedupe_keeps_higher_ratings_count() -> None:
    """Expected use: when same ISBN, higher ratings_count wins."""
    a = make_book("ol_A", "openlibrary", isbn_13="9781234567890", ratings_count=100)
    b = make_book("gb_B", "google_books", isbn_13="9781234567890", ratings_count=999)
    out = to_silver.dedupe([a, b])
    assert len(out) == 1
    assert out[0].book_id == "gb_B"


def test_dedupe_prefers_openlibrary_on_tie() -> None:
    """Edge case: tied ratings_count → Open Library wins (richer subject vocab)."""
    a = make_book("gb_A", "google_books", isbn_13="9781234567890", ratings_count=10)
    b = make_book("ol_B", "openlibrary", isbn_13="9781234567890", ratings_count=10)
    out = to_silver.dedupe([a, b])
    assert len(out) == 1
    assert out[0].source == "openlibrary"


def test_dedupe_falls_back_to_title_author_without_isbn() -> None:
    """Failure case: missing ISBNs still dedup by title+author."""
    a = make_book("ol_A", "openlibrary", isbn_13=None, ratings_count=5)
    b = make_book("gb_B", "google_books", isbn_13=None, ratings_count=50)
    out = to_silver.dedupe([a, b])
    assert len(out) == 1
    assert out[0].book_id == "gb_B"


def test_write_silver_parquet_round_trips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Expected use: written parquet can be read back as polars DataFrame."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    books = [make_book("ol_A", "openlibrary", isbn_13="9781234567890")]
    out = to_silver.write_silver_parquet(books)
    df = pl.read_parquet(out)
    assert df.shape == (1, 14)
    assert df["isbn_13"].to_list() == ["9781234567890"]
    get_settings.cache_clear()
