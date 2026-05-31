"""Extract the genre vocabulary from books_with_genres.parquet.

Writes the fixed genre vocab (10 categories) to a JSON file. The same vocab is
used by all downstream item/user feature pipelines, so it must be deterministic.

Usage:
    .venv/bin/python -m mybookrec.features.generate_vocab
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mybookrec import DATA_DIR, ROOT_DIR

VOCAB_PATH = ROOT_DIR / "mybookrec" / "features" / "genre_vocab.json"


def extract_genre_vocab() -> list[str]:
    """Read the genre struct field names from the source parquet.

    Returns:
        List of genre name strings in the parquet's natural field order.
    """
    schema = pl.read_parquet(DATA_DIR / "transformed" / "shared" / "books_with_genres.parquet").schema
    return [field.name for field in schema["genres"].fields]


def write_vocab(vocab: list[str], output_path: Path = VOCAB_PATH) -> None:
    """Persist the genre vocab as a JSON list.

    Args:
        vocab: Ordered list of genre name strings.
        output_path: Destination JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)


def main() -> None:
    """Build and save the genre vocab."""
    vocab = extract_genre_vocab()
    print(f"Unique genres ({len(vocab)}): {vocab}")
    write_vocab(vocab)
    print(f"Saved to {VOCAB_PATH}")


if __name__ == "__main__":
    main()
