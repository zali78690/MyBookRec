"""Tests for the Google Books adapter (network mocked with respx)."""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
import respx

from mybookrec.ingest import google_books

SAMPLE_ITEM = {
    "id": "abc123",
    "volumeInfo": {
        "title": "The Final Empire",
        "authors": ["Brandon Sanderson"],
        "publishedDate": "2006-07-25",
        "description": "<b>Mistborn</b> book one.",
        "industryIdentifiers": [
            {"type": "ISBN_10", "identifier": "0765311780"},
            {"type": "ISBN_13", "identifier": "9780765311788"},
        ],
        "pageCount": 541,
        "categories": ["Fiction / Fantasy / Epic"],
        "averageRating": 4.5,
        "ratingsCount": 1000,
        "language": "en",
    },
    "saleInfo": {"isEbook": True},
}


def test_parse_year_handles_variable_published_date_formats() -> None:
    """Edge case: publishedDate can be YYYY, YYYY-MM, or YYYY-MM-DD."""
    assert google_books.parse_year("2006") == 2006
    assert google_books.parse_year("2006-07") == 2006
    assert google_books.parse_year("2006-07-25") == 2006
    assert google_books.parse_year(None) is None
    assert google_books.parse_year("garbage") is None


def test_pick_isbn13_prefers_correct_type() -> None:
    """Expected use: only ISBN_13 entries are returned, never ISBN_10."""
    identifiers = [
        {"type": "ISBN_10", "identifier": "0765311780"},
        {"type": "ISBN_13", "identifier": "9780765311788"},
    ]
    assert google_books.pick_isbn13(identifiers) == "9780765311788"


def test_pick_isbn13_returns_none_when_absent() -> None:
    """Edge case: no ISBN_13 entry → None."""
    assert google_books.pick_isbn13([{"type": "ISSN", "identifier": "0000"}]) is None
    assert google_books.pick_isbn13(None) is None


def test_flatten_categories_splits_and_dedupes() -> None:
    """Expected use: hierarchical slash-delimited categories become a flat unique list."""
    cats = ["Fiction / Fantasy / Epic", "Fiction / Adventure"]
    assert google_books.flatten_categories(cats) == ["Fiction", "Fantasy", "Epic", "Adventure"]


def test_adapt_item_strips_html_from_description() -> None:
    """Expected use: HTML in descriptions is removed."""
    book = google_books.adapt_item(SAMPLE_ITEM)
    assert book.description == "Mistborn book one."
    assert book.isbn_13 == "9780765311788"
    assert book.is_ebook is True
    assert book.published_year == 2006
    assert book.genres == ["Fiction", "Fantasy", "Epic"]


def test_adapt_item_raises_without_id() -> None:
    """Failure case: items without `id` raise ValueError."""
    with pytest.raises(ValueError):
        google_books.adapt_item({"volumeInfo": {"title": "x"}})


def install_test_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, api_key: str | None) -> None:
    """Replace `get_settings` with one that bypasses the real .env file.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest tmp directory used as data_dir.
        api_key: Value to install as `google_books_api_key` (None to test the missing-key path).
    """
    from mybookrec import settings as settings_mod

    test_settings = settings_mod.Settings(
        _env_file=None,
        mybookrec_data_dir=tmp_path,
        google_books_api_key=api_key,
    )
    monkeypatch.setattr(settings_mod, "get_settings", lambda: test_settings)
    monkeypatch.setattr("mybookrec.ingest.google_books.get_settings", lambda: test_settings)


def test_fetch_search_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Failure case: missing API key raises a clear RuntimeError BEFORE any HTTP call."""
    install_test_settings(monkeypatch, tmp_path, api_key=None)
    with pytest.raises(RuntimeError, match="GOOGLE_BOOKS_API_KEY"):
        google_books.fetch_search("mistborn")


@respx.mock
def test_fetch_search_writes_bronze_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Expected use: fetch_search appends each item as one JSON line."""
    install_test_settings(monkeypatch, tmp_path, api_key="dummy")
    respx.get(google_books.VOLUMES_URL).mock(return_value=httpx.Response(200, json={"items": [SAMPLE_ITEM]}))

    out = tmp_path / "gb.jsonl"
    items = google_books.fetch_search("mistborn", limit=1, out_path=out)
    assert len(items) == 1
    assert json.loads(out.read_text().strip())["id"] == "abc123"
