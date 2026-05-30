"""Bronze → silver: parse JSONL + dedupe + write parquet.

Bronze layout: `data/bronze/<source>/<YYYY-MM-DD>/<query-slug>.jsonl` (one raw record per line).
Silver layout: `data/silver/books.parquet` (one row per unique book, cross-source).

Dedup rule: prefer ISBN-13 if both records have one; fall back to lowercase title|first-author.
When duplicates exist, Open Library wins (richer subject vocab) unless GB has a higher
`ratings_count` (better recency).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mybookrec.ingest import google_books, openlibrary
from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings


def read_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines and malformed rows.

    Args:
        path: Path to a `.jsonl` file.

    Yields:
        Parsed JSON dict per non-empty line.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_silver_from_bronze(source: str | None = None) -> list[SilverBook]:
    """Walk the bronze tree and return all SilverBook records produced by the matching adapters.

    Args:
        source: Restrict to one source ("openlibrary" or "google_books"); None = both.

    Returns:
        Flat list of SilverBook records (may contain cross-source duplicates).
    """
    settings = get_settings()
    bronze_root = settings.bronze_dir
    books: list[SilverBook] = []

    if source in (None, "openlibrary"):
        for f in sorted(bronze_root.glob("openlibrary/**/*.jsonl")):
            books.extend(openlibrary.to_silver(read_jsonl(f)))

    if source in (None, "google_books"):
        for f in sorted(bronze_root.glob("google_books/**/*.jsonl")):
            books.extend(google_books.to_silver(read_jsonl(f)))

    return books


def dedupe(books: Iterable[SilverBook]) -> list[SilverBook]:
    """Drop duplicates by `dedup_key`. Keep richer or higher-ratings_count record on conflict.

    Args:
        books: Iterable of SilverBook records, possibly cross-source.

    Returns:
        Deduplicated list.
    """
    keep: dict[str, SilverBook] = {}
    for book in books:
        key = book.dedup_key()
        existing = keep.get(key)
        if existing is None or prefer(book, existing):
            keep[key] = book
    return list(keep.values())


def prefer(candidate: SilverBook, existing: SilverBook) -> bool:
    """Return True if `candidate` should replace `existing` in the dedup map.

    Tie-break: more ratings wins; on equal ratings, Open Library wins (broader subject vocab).

    Args:
        candidate: New book record being considered.
        existing: Already-kept book under the same dedup key.

    Returns:
        True if candidate should overwrite existing.
    """
    c_ratings = candidate.ratings_count or 0
    e_ratings = existing.ratings_count or 0
    if c_ratings != e_ratings:
        return c_ratings > e_ratings
    return candidate.source == "openlibrary" and existing.source != "openlibrary"


def write_silver_parquet(books: Iterable[SilverBook], out_path: Path | None = None) -> Path:
    """Serialise SilverBooks as a parquet file.

    Args:
        books: Records to write.
        out_path: Override destination. Defaults to `data/silver/books.parquet`.

    Returns:
        Path written.
    """
    settings = get_settings()
    if out_path is None:
        out_path = settings.silver_dir / "books.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [b.model_dump() for b in books]
    df = pl.DataFrame(rows)
    df.write_parquet(out_path, compression="zstd")
    return out_path


def run(source: str | None = None) -> tuple[Path, int]:
    """Full bronze→silver pass for one or both sources.

    Args:
        source: Restrict to one source; None = both.

    Returns:
        Tuple of (parquet_path, n_rows_written).
    """
    books = dedupe(collect_silver_from_bronze(source))
    out = write_silver_parquet(books)
    return out, len(books)
