"""Tests for the Open Library bulk-dump loader."""

from __future__ import annotations

import gzip
import json
import pathlib

import pytest

from mybookrec.ingest import bulk_openlibrary
from mybookrec.settings import get_settings


def write_fake_dump(path: pathlib.Path, records: list[dict]) -> None:
    """Write OL-dump-formatted lines (5 tab-separated cols) gzipped to `path`."""
    with gzip.open(path, "wb") as f:
        for r in records:
            line = f"/type/work\t{r['key']}\t1\t2024-01-01\t{json.dumps(r)}\n"
            f.write(line.encode("utf-8"))


def test_parse_dump_line_returns_record_for_well_formed_line() -> None:
    """Expected use: a 5-column line parses to its JSON payload."""
    line = b"/type/work\t/works/OL1W\t3\t2024-01-01\t" + json.dumps({"title": "Hi"}).encode("utf-8")
    record = bulk_openlibrary.parse_dump_line(line)
    assert record == {"title": "Hi"}


def test_parse_dump_line_drops_lines_with_wrong_column_count() -> None:
    """Edge case: lines without 5 tabs return None."""
    assert bulk_openlibrary.parse_dump_line(b"only\ttwo") is None
    assert bulk_openlibrary.parse_dump_line(b"") is None


def test_parse_dump_line_drops_malformed_json() -> None:
    """Failure case: invalid JSON in the last column returns None."""
    line = b"/type/work\t/works/OL1W\t3\t2024-01-01\t{not json}"
    assert bulk_openlibrary.parse_dump_line(line) is None


def test_is_keepable_work_requires_title_subjects_description() -> None:
    """Expected + edge: keep filter requires title + subjects + description."""
    keep = {"title": "T", "subjects": ["A"], "description": "D"}
    assert bulk_openlibrary.is_keepable_work(keep) is True
    assert bulk_openlibrary.is_keepable_work({"subjects": ["A"], "description": "D"}) is False
    assert bulk_openlibrary.is_keepable_work({"title": "T", "description": "D"}) is False
    assert bulk_openlibrary.is_keepable_work({"title": "T", "subjects": ["A"]}) is False


def test_is_keepable_work_respects_min_year_from_publish_date() -> None:
    """Expected use: min_year drops records with explicit first_publish_date before the cutoff."""
    base = {"title": "T", "subjects": ["A"], "description": "D"}
    assert bulk_openlibrary.is_keepable_work({**base, "first_publish_date": "2020"}, min_year=2018) is True
    assert bulk_openlibrary.is_keepable_work({**base, "first_publish_date": "2015"}, min_year=2018) is False


def test_is_keepable_work_falls_back_to_created_year() -> None:
    """Edge case: when first_publish_date is missing, created date is the proxy.

    Reflects reality: most OL works lack first_publish_date. Without this fallback the
    year filter would drop ~all records.
    """
    base = {"title": "T", "subjects": ["A"], "description": "D"}
    created_recent = {"type": "/type/datetime", "value": "2023-04-12T10:00:00"}
    created_old = {"type": "/type/datetime", "value": "2015-06-01T00:00:00"}
    assert bulk_openlibrary.is_keepable_work({**base, "created": created_recent}, min_year=2018) is True
    assert bulk_openlibrary.is_keepable_work({**base, "created": created_old}, min_year=2018) is False
    # Records with no year at all fail closed.
    assert bulk_openlibrary.is_keepable_work(base, min_year=2018) is False


def test_extract_year_from_text_handles_variable_formats() -> None:
    """Edge case: dates arrive in many shapes; we want a year out of each plausible one."""
    assert bulk_openlibrary.extract_year_from_text("2020") == 2020
    assert bulk_openlibrary.extract_year_from_text("2020-03-15") == 2020
    assert bulk_openlibrary.extract_year_from_text("2020-03-15T12:00:00") == 2020
    assert bulk_openlibrary.extract_year_from_text("March 2020") == 2020
    assert bulk_openlibrary.extract_year_from_text("Spring 2020") == 2020
    assert bulk_openlibrary.extract_year_from_text(None) is None
    assert bulk_openlibrary.extract_year_from_text("no year here") is None
    # Implausible years are rejected.
    assert bulk_openlibrary.extract_year_from_text("9999-01-01") is None


def test_get_datetime_value_unwraps_ol_wrapper() -> None:
    """Edge case: OL's `{type, value}` shape unwraps to the inner ISO string."""
    assert (
        bulk_openlibrary.get_datetime_value({"type": "/type/datetime", "value": "2023-01-01T00:00:00"})
        == "2023-01-01T00:00:00"
    )
    assert bulk_openlibrary.get_datetime_value("2023-01-01") == "2023-01-01"
    assert bulk_openlibrary.get_datetime_value(None) is None
    assert bulk_openlibrary.get_datetime_value(42) is None


def test_adapt_work_extracts_author_keys_as_grouping_ids() -> None:
    """Expected use: dump's author key (e.g. OL26320A) becomes the SilverBook authors[0]."""
    record = {
        "key": "/works/OL1234W",
        "title": "Mistborn",
        "subjects": ["Fantasy"],
        "description": "epic novel",
        "authors": [{"author": {"key": "/authors/OL26320A"}}],
    }
    silver = bulk_openlibrary.adapt_work(record)
    assert silver.book_id == "ol_OL1234W"
    assert silver.authors == ["OL26320A"]
    assert silver.description == "epic novel"
    assert silver.genres == ["Fantasy"]
    assert silver.isbn_13 is None  # works dump has no ISBN


def test_adapt_work_unwraps_dict_description() -> None:
    """Edge case: OL `description: {type, value}` form is unwrapped by SilverBook."""
    record = {
        "key": "/works/OL1W",
        "title": "T",
        "subjects": ["S"],
        "description": {"type": "/type/text", "value": "actual"},
    }
    silver = bulk_openlibrary.adapt_work(record)
    assert silver.description == "actual"


def test_adapt_work_raises_without_key() -> None:
    """Failure case: missing /works key → ValueError."""
    with pytest.raises(ValueError):
        bulk_openlibrary.adapt_work({"title": "no key"})


def test_run_shards_and_writes_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Expected use: kept records shard into multiple JSONLs and a manifest is written."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    dump = tmp_path / "fake.txt.gz"
    records = [
        {
            "key": f"/works/OL{i}W",
            "title": f"Book {i}",
            "subjects": ["Fiction"],
            "description": f"desc {i}",
            "authors": [{"author": {"key": f"/authors/OL{i}A"}}],
        }
        for i in range(7)
    ]
    write_fake_dump(dump, records)

    out_dir, n_kept = bulk_openlibrary.run(
        input_path=dump,
        url="",
        limit=None,
        shard_size=3,
        tag="unit_test",
    )
    assert n_kept == 7
    shards = sorted(out_dir.glob("works_*.jsonl"))
    assert len(shards) >= 2  # 7 records / shard_size 3 → 3 shards
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_kept"] == 7
    assert manifest["n_scanned"] == 7
    get_settings.cache_clear()


def test_run_respects_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Edge case: --limit stops the loop early."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    dump = tmp_path / "fake.txt.gz"
    records = [
        {
            "key": f"/works/OL{i}W",
            "title": f"Book {i}",
            "subjects": ["Fiction"],
            "description": f"desc {i}",
        }
        for i in range(100)
    ]
    write_fake_dump(dump, records)

    _, n_kept = bulk_openlibrary.run(input_path=dump, url="", limit=5, shard_size=10, tag="limit_test")
    assert n_kept == 5
    get_settings.cache_clear()
