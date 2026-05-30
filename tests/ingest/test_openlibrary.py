"""Tests for the Open Library adapter (network mocked with respx)."""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
import respx

from mybookrec.ingest import openlibrary

SAMPLE_DOC = {
    "key": "/works/OL5738148W",
    "title": "The Final Empire",
    "author_name": ["Brandon Sanderson"],
    "author_key": ["OL2480350A"],
    "isbn": ["1234567890", "9781234567890"],
    "language": ["eng"],
    "first_publish_year": 2006,
    "number_of_pages_median": 541,
    "ratings_average": 4.42,
    "ratings_count": 3000,
    "ebook_access": "public",
    "subject": ["Fantasy", "Magic", "Fiction"],
    "description": "Plain string description",
}


def test_adapt_doc_produces_expected_silver_book() -> None:
    """Expected use: a typical search doc maps cleanly to SilverBook."""
    book = openlibrary.adapt_doc(SAMPLE_DOC)
    assert book.book_id == "ol_OL5738148W"
    assert book.source == "openlibrary"
    assert book.title == "The Final Empire"
    assert book.authors == ["Brandon Sanderson"]
    assert book.genres == ["Fantasy", "Magic", "Fiction"]
    assert book.language == "en"
    assert book.published_year == 2006
    assert book.num_pages == 541
    assert book.average_rating == pytest.approx(4.42)
    assert book.is_ebook is True
    assert book.isbn_13 == "9781234567890"


def test_adapt_doc_raises_without_key() -> None:
    """Failure case: missing `key` field raises ValueError."""
    with pytest.raises(ValueError):
        openlibrary.adapt_doc({"title": "no key"})


def test_to_silver_skips_unparseable_rows() -> None:
    """Edge case: bad rows are dropped silently instead of raising."""
    docs = [SAMPLE_DOC, {"title": "no key"}, SAMPLE_DOC]
    silver_books = list(openlibrary.to_silver(docs))
    assert len(silver_books) == 2


@respx.mock
def test_fetch_search_writes_bronze_jsonl(tmp_path: pathlib.Path) -> None:
    """Expected use: fetch_search writes raw docs to the requested path."""
    respx.get(openlibrary.SEARCH_URL).mock(return_value=httpx.Response(200, json={"docs": [SAMPLE_DOC, SAMPLE_DOC]}))
    out = tmp_path / "ol.jsonl"
    docs = openlibrary.fetch_search("mistborn", limit=2, out_path=out)
    assert len(docs) == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["key"] == "/works/OL5738148W"


@respx.mock
def test_fetch_search_handles_empty_response() -> None:
    """Edge case: when API returns no docs, we get an empty list (not an exception)."""
    respx.get(openlibrary.SEARCH_URL).mock(return_value=httpx.Response(200, json={"docs": []}))
    docs = openlibrary.fetch_search("nothingmatches")
    assert docs == []
