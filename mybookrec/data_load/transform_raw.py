"""Transform raw UCSD JSON dumps + personal CSV → cleaned parquets.

Pipeline:
    1. Stream books JSON, cast columns, filter ratings_count > 5.
    2. Stream genres JSON, inner-join onto books.
    3. Save books_with_genres.parquet.
    4. Stream interactions JSON, cast rating + date_added, filter rating > 0.
    5. Inner-join interactions onto books, time-aware per-user 70/20/10 split.
    6. Save books_with_interactions.parquet.
    7. Transform personal Goodreads CSV export → my_books.csv.

Usage:
    .venv/bin/python -m mybookrec.data_load.transform_raw
"""

from __future__ import annotations

import polars as pl

from mybookrec import DATA_DIR


def load_books_lazy() -> pl.LazyFrame:
    """Lazy-load the raw books JSON with type casts and the ratings_count filter.

    Returns:
        LazyFrame ready to join against genres/interactions.
    """
    return (
        pl.scan_ndjson(DATA_DIR / "raw" / "ucsd" / "goodreads_books.json.gz", low_memory=True)
        .select(
            pl.col("book_id"),
            pl.col("num_pages").cast(pl.Int64, strict=False),
            pl.col("is_ebook") == True,  # noqa: E712 — polars expression, not Python boolean comparison
            pl.col("average_rating").cast(pl.Float64, strict=False),
            pl.col("ratings_count").cast(pl.Int64, strict=False),
            pl.col("description"),
            pl.col("title"),
        )
        .filter(pl.col("ratings_count") > 5)
    )


def load_genres_lazy() -> pl.LazyFrame:
    """Lazy-load the raw genres JSON.

    Returns:
        LazyFrame with (book_id, genres) columns.
    """
    return pl.scan_ndjson(DATA_DIR / "raw" / "ucsd" / "goodreads_book_genres_initial.json", low_memory=True).select(
        pl.col("book_id"), pl.col("genres")
    )


def load_interactions_lazy() -> pl.LazyFrame:
    """Lazy-load the raw interactions JSON with date parsing and rating filter.

    Returns:
        LazyFrame with (user_id, book_id, rating, date_added) where rating > 0.
    """
    return (
        pl.scan_ndjson(DATA_DIR / "raw" / "ucsd" / "goodreads_interactions_dedup.json.gz", low_memory=True)
        .select(
            pl.col("user_id"),
            pl.col("book_id"),
            pl.col("rating").cast(pl.Float64, strict=False),
            pl.col("date_added").str.replace_all(r"\s+", " ").str.to_datetime("%a %b %d %H:%M:%S %z %Y", strict=False),
        )
        .filter(pl.col("rating") > 0)
    )


def save_personal_my_books() -> None:
    """Transform the personal Goodreads CSV export → slim my_books.csv."""
    src = DATA_DIR / "raw" / "personal" / "goodreads_library_export.csv"
    dst = DATA_DIR / "transformed" / "shared" / "my_books.csv"
    (
        pl.read_csv(src)
        .select(
            pl.col("Book Id").alias("book_id").cast(pl.Int64, strict=False),
            pl.col("My Rating").alias("my_rating").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("my_rating") > 0)
        .write_csv(dst)
    )
    print(f"Saved {dst.name}")


def save_books_with_genres(books: pl.LazyFrame, genres: pl.LazyFrame) -> None:
    """Inner-join books with genres, drop genre-less books, write parquet."""
    output = DATA_DIR / "transformed" / "shared" / "books_with_genres.parquet"
    books.join(genres, on="book_id", how="inner").sink_parquet(output)
    print(f"Saved {output.name}")


def save_books_with_interactions(books: pl.LazyFrame, interactions: pl.LazyFrame) -> None:
    """Inner-join books with interactions, apply per-user temporal split, write parquet.

    Per-user dense-rank by date_added then bucket into train (first 70%),
    test (next 20%), validation (last 10%). Per-user splitting prevents leakage.
    """
    output = DATA_DIR / "transformed" / "shared" / "books_with_interactions.parquet"
    (
        books.join(interactions, on="book_id", how="inner")
        .with_columns(pl.col("date_added").rank("dense", descending=False).over("user_id").alias("interaction_rank"))
        .with_columns(
            pl.when(pl.col("interaction_rank") <= pl.col("interaction_rank").max().over("user_id") * 0.7)
            .then(pl.lit("train"))
            .when(pl.col("interaction_rank") <= pl.col("interaction_rank").max().over("user_id") * 0.9)
            .then(pl.lit("test"))
            .otherwise(pl.lit("validation"))
            .alias("data_split")
        )
        .sink_parquet(output)
    )
    print(f"Saved {output.name}")


def main() -> None:
    """Run the full raw → cleaned-parquet pipeline."""
    (DATA_DIR / "transformed" / "shared").mkdir(parents=True, exist_ok=True)
    books = load_books_lazy()
    genres = load_genres_lazy()
    interactions = load_interactions_lazy()

    save_personal_my_books()
    save_books_with_genres(books, genres)
    save_books_with_interactions(books, interactions)


if __name__ == "__main__":
    main()
