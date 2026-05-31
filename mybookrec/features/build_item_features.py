"""Build the item-side feature matrices for the v1 feature set.

Produces three artifacts under data/transformed/:
- genre_matrix.npy: (n_books, n_genres) L2-normalized count-weighted genre vectors.
- num_pages_normalized.npy: (n_books,) page counts normalized to [0, 1] via
  percentile-clipped min-max with median imputation.
- num_pages_norm_params.json: {p1, p99, median} for reuse at inference time.

The book_embeddings.npy file comes from scripts/embed.ipynb (Colab GPU) — this
script does not produce it.

Usage:
    .venv/bin/python -m mybookrec.features.build_item_features
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from mybookrec import DATA_DIR, ROOT_DIR


def load_vocab_and_books() -> tuple[list[str], pl.DataFrame]:
    """Load the genre vocab and the books_with_genres parquet, verifying schema match.

    Returns:
        Tuple of (genre vocab list, books DataFrame).

    Raises:
        AssertionError: If the parquet's genre struct fields don't equal the vocab.
    """
    books = pl.read_parquet(DATA_DIR / "transformed" / "shared" / "books_with_genres.parquet")
    with open(ROOT_DIR / "mybookrec" / "features" / "genre_vocab.json") as f:
        vocab = json.load(f)

    struct_fields = set(books["genres"].struct.fields)
    vocab_set = set(vocab)
    assert struct_fields == vocab_set, (
        f"Schema mismatch.\n"
        f"  in struct, not vocab: {struct_fields - vocab_set}\n"
        f"  in vocab, not struct: {vocab_set - struct_fields}"
    )
    print(f"Loaded {len(books):,} books; {len(vocab)} genres; schema matches.")
    return vocab, books


def build_genre_matrix(books: pl.DataFrame, vocab: list[str]) -> np.ndarray:
    """Build a count-weighted L2-normalized (n_books, n_genres) matrix.

    Iterates the vocab (small) rather than per-row over millions of books.

    Args:
        books: DataFrame containing a `genres` struct column.
        vocab: Ordered list of genre names; column order is preserved.

    Returns:
        Float32 matrix shape (n_books, n_genres). All-zero books stay [0, 0, ...]
        (never NaN) thanks to a guarded division.
    """
    n_books = books.shape[0]
    matrix = np.zeros((n_books, len(vocab)), dtype=np.float32)
    for col_idx, genre_name in enumerate(vocab):
        counts = books["genres"].struct.field(genre_name).fill_null(0).to_numpy()
        matrix[:, col_idx] = counts

    zero_rows = int((matrix.sum(axis=1) == 0).sum())
    print(f"Books with all-zero genre counts: {zero_rows:,} / {n_books:,}")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return (matrix / safe_norms).astype(np.float32)


def normalize_pages(books: pl.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize num_pages to [0, 1] via percentile-clipped min-max with imputation.

    Treats 0 and null as missing values, imputes with the median of present values,
    then clips to [p1, p99] before rescaling.

    Args:
        books: DataFrame containing a `num_pages` column.

    Returns:
        Tuple of (normalized_pages float32 vector, {p1, p99, median} params dict).
    """
    pages = books["num_pages"].cast(pl.Float64).to_numpy()
    missing = np.isnan(pages) | (pages <= 0)
    median = float(np.median(pages[~missing]))
    pages = np.where(missing, median, pages)

    p1, p99 = (float(x) for x in np.percentile(pages, [1, 99]))
    clipped = np.clip(pages, p1, p99)
    normalized = ((clipped - p1) / (p99 - p1)).astype(np.float32)

    print(
        f"num_pages: missing={int(missing.sum()):,} ({missing.mean() * 100:.2f}%) "
        f"median={median:.1f} p1={p1:.1f} p99={p99:.1f}"
    )
    print(f"normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    return normalized, {"p1": p1, "p99": p99, "median": median}


def main() -> None:
    """Build and save the genre matrix + normalized pages."""
    vocab, books = load_vocab_and_books()
    shared = DATA_DIR / "transformed" / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    genre_matrix = build_genre_matrix(books, vocab)
    np.save(shared / "genre_matrix.npy", genre_matrix)
    print(f"Saved genre_matrix {genre_matrix.shape} to {shared / 'genre_matrix.npy'}")

    normalized_pages, params = normalize_pages(books)
    np.save(shared / "num_pages_normalized.npy", normalized_pages)
    with open(shared / "num_pages_norm_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Saved normalized pages + params")


if __name__ == "__main__":
    main()
