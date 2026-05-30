"""Open Library fetcher + bronze→silver adapter.

Free, no API key. Search API returns a thin doc by default — we explicitly request `fields=`
to get ratings, pages, subjects, etc. in a single round trip (no `/works/{olid}` follow-up
needed for most books).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mybookrec.ingest.http_client import get_json_with_retry, rate_limited_client
from mybookrec.ingest.language_map import to_iso_639_1
from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings

SEARCH_URL = "https://openlibrary.org/search.json"
SEARCH_FIELDS = (
    "key,title,author_name,author_key,isbn,language,first_publish_year,publish_year,"
    "number_of_pages_median,ratings_average,ratings_count,ebook_access,subject,publisher,description"
)


def fetch_search(
    query: str,
    *,
    limit: int = 50,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch raw Open Library search results and optionally append to a bronze JSONL file.

    Args:
        query: Free-text search query (e.g. an author name, subject, or title).
        limit: Maximum results to retrieve (Open Library caps single requests at 100).
        out_path: If set, append each doc as one JSON line to this file.

    Returns:
        List of raw `docs[i]` dicts as returned by the API.
    """
    settings = get_settings()
    headers = {"User-Agent": settings.openlibrary_user_agent, "Accept": "application/json"}
    with rate_limited_client(
        rate_per_sec=settings.ingest_rate_limit_per_sec,
        timeout_sec=settings.ingest_request_timeout_sec,
        headers=headers,
    ) as (client, bucket):
        bucket.acquire()
        payload = get_json_with_retry(
            client,
            SEARCH_URL,
            params={"q": query, "limit": min(limit, 100), "fields": SEARCH_FIELDS},
        )
    docs: list[dict[str, Any]] = payload.get("docs", []) if isinstance(payload, dict) else []
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return docs


def to_silver(docs: Iterable[dict[str, Any]]) -> Iterator[SilverBook]:
    """Convert raw Open Library docs to SilverBook records, skipping unparseable rows.

    Args:
        docs: Iterable of `docs[i]` dicts from the Search API.

    Yields:
        SilverBook for each doc that has at minimum a key and a title.
    """
    for doc in docs:
        try:
            yield adapt_doc(doc)
        except Exception:
            # We'd rather drop a malformed row than fail the whole batch — bronze is the source
            # of truth, silver is reproducible.
            continue


def adapt_doc(doc: dict[str, Any]) -> SilverBook:
    """Convert one Open Library search doc into a SilverBook.

    Args:
        doc: A single `docs[i]` dict.

    Returns:
        A validated SilverBook.

    Raises:
        ValueError: If the doc lacks a `key`.
    """
    key = doc.get("key", "")
    olid = key.rsplit("/", 1)[-1] if key else ""
    if not olid:
        raise ValueError("missing key")

    isbns = [s for s in doc.get("isbn", []) if isinstance(s, str) and len(s.replace("-", "")) == 13]
    isbn_13 = isbns[0] if isbns else None

    languages_raw = doc.get("language") or []
    language = to_iso_639_1(languages_raw[0]) if languages_raw else None

    ebook_access = doc.get("ebook_access")
    is_ebook = ebook_access != "no_ebook" if ebook_access is not None else None

    return SilverBook(
        book_id=f"ol_{olid}",
        raw_id=olid,
        source="openlibrary",
        title=doc.get("title", ""),
        description=doc.get("description"),
        num_pages=doc.get("number_of_pages_median"),
        average_rating=doc.get("ratings_average"),
        ratings_count=doc.get("ratings_count"),
        is_ebook=is_ebook,
        authors=list(doc.get("author_name") or []),
        genres=list(doc.get("subject") or []),
        language=language,
        published_year=doc.get("first_publish_year"),
        isbn_13=isbn_13,
    )
