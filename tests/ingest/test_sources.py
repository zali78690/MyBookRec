"""Tests for the BookSource registry."""

from __future__ import annotations

import pytest

from mybookrec.ingest import bulk_openlibrary, google_books, openlibrary
from mybookrec.ingest.sources import SOURCES, filter_sources, source_names


def test_registry_contains_all_three_current_sources() -> None:
    """Expected use: every source we've shipped is registered with the right adapter."""
    by_name = {s.name: s for s in SOURCES}
    assert by_name["openlibrary"].to_silver is openlibrary.to_silver
    assert by_name["openlibrary_dump"].to_silver is bulk_openlibrary.to_silver
    assert by_name["google_books"].to_silver is google_books.to_silver


def test_each_source_has_a_distinct_bronze_glob() -> None:
    """Edge case: globs must be distinct so silver doesn't double-count records."""
    globs = [s.bronze_glob for s in SOURCES]
    assert len(globs) == len(set(globs))


def test_filter_sources_returns_all_when_name_is_none() -> None:
    """Expected use: passing None means 'every registered source'."""
    assert filter_sources(None) == SOURCES


def test_filter_sources_returns_only_the_matching_source() -> None:
    """Expected use: passing a name restricts to that source."""
    matches = filter_sources("openlibrary")
    assert len(matches) == 1
    assert matches[0].name == "openlibrary"


def test_filter_sources_raises_on_unknown_name() -> None:
    """Failure case: an unknown source name fails loudly so CLI mistakes are visible."""
    with pytest.raises(ValueError, match="Unknown source"):
        filter_sources("not_a_real_source")


def test_source_names_returns_registered_names_in_order() -> None:
    """Edge case: source_names() reflects registry order — used to drive CLI choices."""
    names = source_names()
    assert names == tuple(s.name for s in SOURCES)
