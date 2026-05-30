"""Build the ISBN-13 → UCSD book_id lookup used by ingest's v4 author fallback.

Walks `data/raw/goodreads_books.json.gz` once and emits
`data/transformed/isbn13_to_book_id.json` containing
`{isbn_13_digits_only: book_id}` for every UCSD book that has a valid 13-digit ISBN.

This is a one-off precompute. New ingested books look up by ISBN-13: if matched,
they're treated as a known UCSD book and can borrow the trained author embedding.
If unmatched, the gold pipeline falls back to a cold-start embedding.

Usage:
    .venv/bin/python -m mybookrec.features.build_isbn_index
"""

from __future__ import annotations

import gzip
import json
import re
import time

from mybookrec.settings import get_settings

DIGIT_RE = re.compile(r"\D")


def normalise_isbn(value: object) -> str | None:
    """Strip non-digits and keep only 13-digit strings.

    Args:
        value: Anything; only strings are considered.

    Returns:
        13-digit ISBN string, or None if the input doesn't yield one.
    """
    if not isinstance(value, str):
        return None
    digits = DIGIT_RE.sub("", value)
    return digits if len(digits) == 13 else None


def main() -> None:
    """Read raw books, write the ISBN-13 → book_id JSON map."""
    settings = get_settings()
    raw_path = settings.raw_dir / "goodreads_books.json.gz"
    out_path = settings.transformed_dir / "isbn13_to_book_id.json"

    mapping: dict[str, str] = {}
    n_total = 0
    t_start = time.time()
    with gzip.open(raw_path, "rt") as f:
        for line in f:
            n_total += 1
            book = json.loads(line)
            isbn = normalise_isbn(book.get("isbn13"))
            book_id = book.get("book_id")
            if isbn and book_id:
                mapping[isbn] = str(book_id)
            if n_total % 100_000 == 0:
                print(f"  [{time.time() - t_start:5.1f}s] scanned {n_total:,}, kept {len(mapping):,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"Wrote {len(mapping):,} ISBN-13 → book_id pairs ({100 * len(mapping) / n_total:.1f}% of books) → {out_path}")


if __name__ == "__main__":
    main()
