"""Tests for the ingestion Pydantic schemas."""

from __future__ import annotations

import pytest

from mybookrec.ingest.schemas import SilverBook, strip_html


def test_strip_html_removes_tags_and_collapses_whitespace() -> None:
    """Expected use: HTML markup in Google Books descriptions is cleaned."""
    text = "<b>Hello</b>   <i>world</i><br/>"
    assert strip_html(text) == "Hello   world"


def test_strip_html_passes_through_none() -> None:
    """Edge case: None in, None out (descriptions are often missing)."""
    assert strip_html(None) is None


def test_silver_book_accepts_dict_description_form() -> None:
    """Open Library returns description as either a str OR a {type, value} dict."""
    book = SilverBook(
        book_id="ol_OL1W",
        raw_id="OL1W",
        source="openlibrary",
        title="Test",
        description={"type": "/type/text", "value": "<p>actual description</p>"},
    )
    assert book.description == "actual description"


def test_silver_book_drops_invalid_isbn() -> None:
    """Failure case: non-13-digit ISBNs are silently nulled rather than mislabelled."""
    book = SilverBook(
        book_id="ol_OL1W",
        raw_id="OL1W",
        source="openlibrary",
        title="Test",
        isbn_13="123",
    )
    assert book.isbn_13 is None


def test_silver_book_accepts_hyphenated_isbn() -> None:
    """Hyphens in ISBN-13 are stripped before length check."""
    book = SilverBook(
        book_id="ol_OL1W",
        raw_id="OL1W",
        source="openlibrary",
        title="Test",
        isbn_13="978-1-2345-6789-0",
    )
    assert book.isbn_13 == "9781234567890"


def test_silver_book_dedup_key_prefers_isbn() -> None:
    """When ISBN-13 is present, dedup key is isbn:<digits>."""
    book = SilverBook(
        book_id="gb_xyz",
        raw_id="xyz",
        source="google_books",
        title="The Final Empire",
        authors=["Brandon Sanderson"],
        isbn_13="9781234567890",
    )
    assert book.dedup_key() == "isbn:9781234567890"


def test_silver_book_dedup_key_falls_back_to_title_author() -> None:
    """When no ISBN, dedup key is ta:<title_lower>|<first_author_lower>."""
    book = SilverBook(
        book_id="gb_xyz",
        raw_id="xyz",
        source="google_books",
        title="The Final Empire",
        authors=["Brandon Sanderson"],
    )
    assert book.dedup_key() == "ta:the final empire|brandon sanderson"


def test_silver_book_rejects_invalid_rating() -> None:
    """Failure case: average_rating must be 0-5."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        SilverBook(
            book_id="ol_X",
            raw_id="X",
            source="openlibrary",
            title="Test",
            average_rating=7.5,
        )
