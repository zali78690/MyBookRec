"""Tests for ISO 639 language code normalisation."""

from __future__ import annotations

from mybookrec.ingest.language_map import to_iso_639_1


def test_two_letter_codes_pass_through_unchanged() -> None:
    """Expected use: Google Books's 2-letter codes are already correct."""
    assert to_iso_639_1("en") == "en"
    assert to_iso_639_1("fr") == "fr"


def test_three_letter_codes_map_to_two_letter() -> None:
    """Expected use: Open Library's 3-letter codes get mapped to 2-letter."""
    assert to_iso_639_1("eng") == "en"
    assert to_iso_639_1("spa") == "es"
    assert to_iso_639_1("jpn") == "ja"


def test_unknown_codes_pass_through() -> None:
    """Edge case: unmapped codes return unchanged — downstream filters drop non-English."""
    assert to_iso_639_1("xyz") == "xyz"


def test_none_returns_none() -> None:
    """Failure case: None propagates."""
    assert to_iso_639_1(None) is None
