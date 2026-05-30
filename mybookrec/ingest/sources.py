"""Registry of book-ingest sources.

Each `BookSource` declares its name, the bronze glob pattern it produces, and the
adapter that converts its raw JSON dicts into `SilverBook` records. The silver
pipeline iterates the registry instead of hard-coding `if source == "openlibrary"`
branches, so adding source #4 is one new module + one registration line.

The fetch mechanism deliberately lives outside this seam — each source's HTTP /
streaming logic is too different to share. What IS shared is the silver
conversion contract: every source produces `Iterator[SilverBook]` from
`Iterable[dict]`. That's the part this registry codifies.

Adding a new source:

    # in mybookrec/ingest/my_new_source.py
    def to_silver(records: Iterable[dict]) -> Iterator[SilverBook]: ...

    # in mybookrec/ingest/sources.py
    BookSource(
        name="my_new_source",
        bronze_glob="my_new_source/**/*.jsonl",
        to_silver=my_new_source.to_silver,
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from mybookrec.ingest import bulk_openlibrary, google_books, openlibrary
from mybookrec.ingest.schemas import SilverBook

SilverAdapter = Callable[[Iterable[dict]], Iterator[SilverBook]]


@dataclass(frozen=True)
class BookSource:
    """One ingest source: name + bronze layout + silver adapter."""

    name: str
    bronze_glob: str
    to_silver: SilverAdapter


SOURCES: tuple[BookSource, ...] = (
    BookSource(
        name="openlibrary",
        bronze_glob="openlibrary/**/*.jsonl",
        to_silver=openlibrary.to_silver,
    ),
    BookSource(
        name="openlibrary_dump",
        bronze_glob="openlibrary_dump/**/works_*.jsonl",
        to_silver=bulk_openlibrary.to_silver,
    ),
    BookSource(
        name="google_books",
        bronze_glob="google_books/**/*.jsonl",
        to_silver=google_books.to_silver,
    ),
)


def source_names() -> tuple[str, ...]:
    """Return all registered source names in registry order.

    Returns:
        Tuple of source name strings, suitable for CLI `choices=` lists.
    """
    return tuple(s.name for s in SOURCES)


def filter_sources(name: str | None) -> tuple[BookSource, ...]:
    """Return the sources matching a name filter (or all, when name is None).

    Args:
        name: A registered source name, or None to mean "all sources".

    Returns:
        The matching sources in registry order.

    Raises:
        ValueError: If `name` is not None and doesn't match any registered source.
    """
    if name is None:
        return SOURCES
    matches = tuple(s for s in SOURCES if s.name == name)
    if not matches:
        raise ValueError(f"Unknown source {name!r}. Registered: {source_names()}")
    return matches
