"""Tests for the embedding-input catalog merger."""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pytest

from mybookrec.features import build_embedding_input
from mybookrec.io.artifacts import TransformedArtifacts
from mybookrec.settings import get_settings


def write_ucsd(tmp_path: pathlib.Path) -> None:
    """Write a tiny synthetic books_with_genres.parquet under <tmp>/transformed/shared/."""
    shared = tmp_path / "transformed" / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "book_id": ["1", "2"],
            "title": ["UCSD One", "UCSD Two"],
            "description": ["desc one", None],
        }
    ).write_parquet(shared / "books_with_genres.parquet")
    (shared / "isbn13_to_book_id.json").write_text(json.dumps({"9780000000001": "1"}))


def write_silver(tmp_path: pathlib.Path, rows: list[dict]) -> None:
    """Write a synthetic silver/books.parquet under <tmp>/silver/."""
    silver = tmp_path / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(silver / "books.parquet")


@pytest.fixture
def isolated_data(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Redirect data_dir so the merger reads + writes inside tmp_path."""
    monkeypatch.setenv("MYBOOKREC_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    write_ucsd(tmp_path)
    yield tmp_path
    get_settings.cache_clear()


def test_expected_use_concats_ucsd_and_silver(isolated_data: pathlib.Path) -> None:
    """Expected use: silver rows are appended after UCSD; source column tags the origin."""
    write_silver(
        isolated_data,
        [
            {"book_id": "ol_X", "title": "Silver X", "description": "x", "source": "openlibrary", "isbn_13": None},
        ],
    )
    n_ucsd, n_silver, n_total = build_embedding_input.run()
    assert n_ucsd == 2
    assert n_silver == 1
    assert n_total == 3

    out = pl.read_parquet(isolated_data / "transformed" / "shared" / "embedding_input.parquet")
    assert set(out.columns) == {"book_id", "title", "description", "source"}
    assert out["source"].unique().sort().to_list() == ["silver_openlibrary", "ucsd"]


def test_failure_case_drops_silver_books_with_matching_ucsd_isbn(isolated_data: pathlib.Path) -> None:
    """Failure case for dedup: silver rows whose ISBN-13 matches UCSD are dropped."""
    write_silver(
        isolated_data,
        [
            {
                "book_id": "ol_X",
                "title": "Dup",
                "description": "d",
                "source": "openlibrary",
                "isbn_13": "9780000000001",  # matches UCSD book "1"
            },
            {
                "book_id": "ol_Y",
                "title": "New",
                "description": "n",
                "source": "openlibrary",
                "isbn_13": "9789999999999",
            },
        ],
    )
    _, n_silver, _ = build_embedding_input.run()
    assert n_silver == 1  # the duplicate ISBN was dropped


def test_edge_case_null_isbn_silver_rows_are_kept(isolated_data: pathlib.Path) -> None:
    """Edge case: rows with null ISBN-13 (e.g. OL works dump) are kept — they can't be
    proven duplicates by ISBN. This is the bug we fixed; without it, the bulk dump
    (which has no ISBN at all) would lose all 754k rows.
    """
    write_silver(
        isolated_data,
        [
            {"book_id": f"ol_{i}", "title": f"Bulk {i}", "description": "d", "source": "openlibrary", "isbn_13": None}
            for i in range(100)
        ],
    )
    _, n_silver, _ = build_embedding_input.run()
    assert n_silver == 100


def test_artifact_size_is_reasonable_proxy_via_row_count(isolated_data: pathlib.Path) -> None:
    """Edge case: row count matches what callers will see in the parquet."""
    write_silver(
        isolated_data,
        [
            {"book_id": "ol_a", "title": "A", "description": "d", "source": "openlibrary", "isbn_13": None},
            {"book_id": "ol_b", "title": "B", "description": "d", "source": "openlibrary", "isbn_13": None},
        ],
    )
    _, _, n_total = build_embedding_input.run()
    out = pl.read_parquet(isolated_data / "transformed" / "shared" / "embedding_input.parquet")
    assert len(out) == n_total


# Quiet unused-fixture import warning
_ = TransformedArtifacts
