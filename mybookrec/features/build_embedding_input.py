"""Combine UCSD + silver bulk books into one minimal input for the embedding pass.

The Colab embedding notebook (`scripts/embed.ipynb`) reads this file from Drive and
produces `book_embeddings_v2.npy` for every row, in order. Keeping the schema minimal
(`book_id`, `title`, `description`) keeps the upload small and matches what the
embedding model actually needs.

Inputs:
- `data/transformed/books_with_genres.parquet` — UCSD catalog (~1.78M books). Has
  structured genres (a struct field) — we drop it here.
- `data/silver/books.parquet` — ingested-from-API books (the bulk OL dump + any
  per-query fetches). Genres are list[str] — also dropped here.

Dedup rule: rows are joined on `book_id`. Since UCSD `book_id`s are numeric Goodreads
ids and silver `book_id`s are prefixed (`ol_*`, `gb_*`), they never collide; the silver
side already de-duplicated cross-source via ISBN-13 / title|author. We do an
additional ISBN-13 vs the UCSD ISBN index to drop silver rows already represented in
UCSD (so we don't embed the same book twice).

Output: `data/transformed/embedding_input.parquet` with columns:
    book_id (str), title (str), description (str), source (str: "ucsd" | "silver_*")

The `source` column lets downstream code split features back out per origin without
re-joining anything.

Usage:
    .venv/bin/python -m mybookrec.features.build_embedding_input
"""

from __future__ import annotations

import polars as pl

from mybookrec.io.artifacts import TransformedArtifacts
from mybookrec.settings import get_settings


def load_ucsd_subset(artifacts: TransformedArtifacts) -> pl.DataFrame:
    """Pull (book_id, title, description) from UCSD books, tagging source='ucsd'.

    Args:
        artifacts: Source of the UCSD `books_with_genres.parquet`.

    Returns:
        DataFrame with the four-column embedding schema.
    """
    return (
        pl.scan_parquet(artifacts.path("books_with_genres.parquet"))
        .select(
            pl.col("book_id").cast(pl.String),
            pl.col("title").cast(pl.String),
            pl.col("description").cast(pl.String),
        )
        .with_columns(pl.lit("ucsd").alias("source"))
        .collect()
    )


def load_silver_subset(silver_path) -> pl.DataFrame:
    """Pull (book_id, title, description, source) from silver, prefixing source.

    Args:
        silver_path: Path to the silver `books.parquet`.

    Returns:
        DataFrame with the four-column embedding schema. `source` is prefixed
        ("silver_openlibrary", "silver_google_books") so downstream code can split.
    """
    return (
        pl.scan_parquet(silver_path)
        .select(
            pl.col("book_id").cast(pl.String),
            pl.col("title").cast(pl.String),
            pl.col("description").cast(pl.String),
            pl.concat_str([pl.lit("silver_"), pl.col("source")]).alias("source"),
            pl.col("isbn_13"),
        )
        .collect()
    )


def drop_silver_books_already_in_ucsd(
    silver: pl.DataFrame,
    isbn13_to_book_id: dict[str, str],
) -> pl.DataFrame:
    """Drop silver rows whose ISBN-13 matches a known UCSD book.

    Rows with a null ISBN-13 (e.g. anything from the OL works dump, which has no
    ISBN field) are KEPT — they can't be proven duplicates by ISBN alone.

    Args:
        silver: Silver subset including an `isbn_13` column.
        isbn13_to_book_id: From `TransformedArtifacts.isbn13_to_book_id` — the
            UCSD ISBN-13 set.

    Returns:
        Silver rows whose ISBN is null OR doesn't match UCSD.
    """
    known_isbns = pl.Series("isbn_13", list(isbn13_to_book_id.keys()), dtype=pl.String)
    is_known_dupe = pl.col("isbn_13").is_in(known_isbns).fill_null(False)
    return silver.filter(~is_known_dupe).drop("isbn_13")


def keep_embeddable_rows(combined: pl.DataFrame) -> pl.DataFrame:
    """Drop rows the embedding model can't usefully process (no title AND no description)."""
    return combined.filter(pl.col("title").is_not_null() | pl.col("description").is_not_null())


def run() -> tuple[int, int, int]:
    """Build and write `data/transformed/embedding_input.parquet`.

    Returns:
        Tuple of (n_ucsd, n_silver_kept, n_total) describing the merge result.
    """
    settings = get_settings()
    artifacts = TransformedArtifacts(settings.transformed_dir)

    ucsd = load_ucsd_subset(artifacts)
    silver_path = settings.silver_dir / "books.parquet"
    if silver_path.exists():
        silver = load_silver_subset(silver_path)
        silver = drop_silver_books_already_in_ucsd(silver, artifacts.isbn13_to_book_id)
    else:
        silver = ucsd.head(0).with_columns(pl.lit(None).alias("isbn_13")).drop("isbn_13")

    combined = pl.concat([ucsd, silver], how="vertical")
    combined = keep_embeddable_rows(combined)

    out_path = settings.transformed_dir / "shared" / "embedding_input.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="zstd")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"[embedding_input] UCSD={len(ucsd):,}  silver_kept={len(silver):,}  "
        f"total={len(combined):,}  size={size_mb:.0f} MB  → {out_path}"
    )
    return len(ucsd), len(silver), len(combined)


def main() -> None:
    """CLI entrypoint."""
    run()


if __name__ == "__main__":
    main()
