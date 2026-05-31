"""Build a slim training_interactions parquet from the full join.

The full books_with_interactions.parquet is ~19 GB because it duplicates every
book column per interaction. The slim version keeps only the 4 columns the
training Dataset needs (user_id, book_id, rating, data_split) and zstd-compresses
to roughly 500 MB. Fits in Drive quota and loads ~10x faster.

Usage:
    .venv/bin/python -m mybookrec.data_load.build_training_interactions
"""

from __future__ import annotations

import polars as pl

from mybookrec import DATA_DIR


def build_slim_parquet() -> None:
    """Stream the full interactions parquet → 4-column zstd parquet."""
    shared = DATA_DIR / "transformed" / "shared"
    src = shared / "books_with_interactions.parquet"
    dst = shared / "training_interactions.parquet"

    (pl.scan_parquet(src).select("user_id", "book_id", "rating", "data_split").sink_parquet(dst, compression="zstd"))

    src_size = src.stat().st_size / 1e9
    dst_size = dst.stat().st_size / 1e9
    print(f"Source: {src.name}  {src_size:.2f} GB")
    print(f"Slim:   {dst.name}  {dst_size:.2f} GB  ({dst_size / src_size:.1%} of original)")


def verify_slim_parquet() -> None:
    """Sanity check: total row count and split distribution."""
    slim = pl.read_parquet(DATA_DIR / "transformed" / "shared" / "training_interactions.parquet")
    print(f"\nRows: {len(slim):,}")
    print(slim.head(3))
    print("\nSplit distribution:")
    print(slim.group_by("data_split").len().sort("data_split"))


def main() -> None:
    """Build then verify the slim training interactions parquet."""
    build_slim_parquet()
    verify_slim_parquet()


if __name__ == "__main__":
    main()
