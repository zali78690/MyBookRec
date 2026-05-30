"""Google Books fetcher + bronze→silver adapter.

Requires `GOOGLE_BOOKS_API_KEY` in `.env` (free tier: 1k queries/day per project). API quirks
handled here: HTML in descriptions, variable-length `publishedDate`, ISBN nested in
`industryIdentifiers`, slash-delimited hierarchical `categories`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mybookrec.ingest.http_client import get_json_with_retry, rate_limited_client
from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings

VOLUMES_URL = "https://www.googleapis.com/books/v1/volumes"


def fetch_search(
    query: str,
    *,
    limit: int = 40,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch raw Google Books search results and optionally append to a bronze JSONL file.

    Args:
        query: Free-text search query.
        limit: Maximum results (Google caps `maxResults` at 40 per call; pages above that
            require manual pagination, deferred until needed).
        out_path: If set, append each item as one JSON line to this file.

    Returns:
        List of raw `items[i]` dicts as returned by the API.

    Raises:
        RuntimeError: If `GOOGLE_BOOKS_API_KEY` is not set.
    """
    settings = get_settings()
    if not settings.google_books_api_key:
        raise RuntimeError("GOOGLE_BOOKS_API_KEY not set in .env — Google Books fetch unavailable.")

    headers = {"Accept": "application/json"}
    with rate_limited_client(
        rate_per_sec=settings.ingest_rate_limit_per_sec,
        timeout_sec=settings.ingest_request_timeout_sec,
        headers=headers,
    ) as (client, bucket):
        bucket.acquire()
        payload = get_json_with_retry(
            client,
            VOLUMES_URL,
            params={"q": query, "maxResults": min(limit, 40), "key": settings.google_books_api_key},
        )
    items: list[dict[str, Any]] = payload.get("items", []) if isinstance(payload, dict) else []
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return items


def to_silver(items: Iterable[dict[str, Any]]) -> Iterator[SilverBook]:
    """Convert raw Google Books items to SilverBook records, skipping unparseable rows.

    Args:
        items: Iterable of `items[i]` dicts from the Volumes API.

    Yields:
        SilverBook for each item that has at minimum an id and a title.
    """
    for item in items:
        try:
            yield adapt_item(item)
        except Exception:
            continue


def parse_year(value: Any) -> int | None:
    """Extract the 4-digit year from a Google Books `publishedDate` string.

    Args:
        value: Raw publishedDate (variable length: YYYY, YYYY-MM, YYYY-MM-DD), or any other type.

    Returns:
        Year as int, or None if not parseable.
    """
    if not isinstance(value, str) or len(value) < 4:
        return None
    head = value[:4]
    return int(head) if head.isdigit() else None


def pick_isbn13(identifiers: list[dict[str, Any]] | None) -> str | None:
    """Return the first ISBN_13 identifier from a Google Books `industryIdentifiers` list.

    Args:
        identifiers: The `industryIdentifiers` array (or None).

    Returns:
        The ISBN-13 string if present, else None.
    """
    if not identifiers:
        return None
    for entry in identifiers:
        if isinstance(entry, dict) and entry.get("type") == "ISBN_13":
            ident = entry.get("identifier")
            if isinstance(ident, str):
                return ident
    return None


def flatten_categories(categories: list[str] | None) -> list[str]:
    """Split slash-delimited hierarchical categories into a flat deduped list.

    Args:
        categories: Google Books `volumeInfo.categories` (e.g. ["Fiction / Fantasy / Epic"]).

    Returns:
        Flat list of unique category strings (order-preserving).
    """
    if not categories:
        return []
    seen: list[str] = []
    for cat in categories:
        if not isinstance(cat, str):
            continue
        for part in cat.split(" / "):
            part = part.strip()
            if part and part not in seen:
                seen.append(part)
    return seen


def adapt_item(item: dict[str, Any]) -> SilverBook:
    """Convert one Google Books `items[i]` dict into a SilverBook.

    Args:
        item: A single `items[i]` dict from the Volumes API.

    Returns:
        A validated SilverBook.

    Raises:
        ValueError: If the item lacks an `id`.
    """
    gb_id = item.get("id")
    if not gb_id:
        raise ValueError("missing id")
    info = item.get("volumeInfo") or {}
    sale = item.get("saleInfo") or {}

    return SilverBook(
        book_id=f"gb_{gb_id}",
        raw_id=gb_id,
        source="google_books",
        title=info.get("title", ""),
        description=info.get("description"),
        num_pages=info.get("pageCount"),
        average_rating=info.get("averageRating"),
        ratings_count=info.get("ratingsCount"),
        is_ebook=sale.get("isEbook"),
        authors=list(info.get("authors") or []),
        genres=flatten_categories(info.get("categories")),
        language=info.get("language"),
        published_year=parse_year(info.get("publishedDate")),
        isbn_13=pick_isbn13(info.get("industryIdentifiers")),
    )
