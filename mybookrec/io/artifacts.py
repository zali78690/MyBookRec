"""Typed lazy loaders for everything under data/transformed/.

One module owns the on-disk filenames + deserialisation conventions for the
artifacts the training pipeline produces. Callers (serve, recommend, ingest,
eval) stop knowing filenames; they ask the registry for a named artifact.

Each loader is a `functools.cached_property`, so:
- The first access pays the IO cost (JSON parse, np.load).
- Subsequent accesses on the same instance are free.
- Tests can swap in a synthetic instance without touching the filesystem.

Constructing the registry is cheap; pass an instance into anything that needs
artifacts and let it cache as needed:

    artifacts = TransformedArtifacts(settings.transformed_dir)
    mapping = artifacts.book_id_to_index
    matrix = artifacts.author_embeddings

For mutation (the FAISS-refresh path extends the book_id mapping), call
`save_book_id_to_index` — it writes to disk AND invalidates the cached value
so the next read sees the update.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

BOOKS_META_COLUMNS: tuple[str, ...] = (
    "book_id",
    "title",
    "num_pages",
    "average_rating",
    "is_ebook",
    "genres",
)


class TransformedArtifacts:
    """Lazy loader for every file the training pipeline writes to data/transformed/.

    Constructor is cheap; properties load on first access and cache thereafter.
    Inject this into modules that need artifacts so they don't need to know
    filenames or how each artifact deserialises.
    """

    def __init__(self, base_dir: Path) -> None:
        """Bind to a transformed-data root directory.

        Args:
            base_dir: Usually `settings.transformed_dir`. Tests pass a tmp_path.
        """
        self.base_dir = Path(base_dir)

    def path(self, filename: str) -> Path:
        """Resolve an artifact's path without loading it.

        Args:
            filename: Name relative to the transformed root (e.g. "book_id_to_index.json").

        Returns:
            Absolute path under `base_dir`.
        """
        return self.base_dir / filename

    @cached_property
    def book_id_to_index(self) -> dict[str, int]:
        """UCSD book_id string → row index in the item matrices.

        Returns:
            Mapping loaded from data/transformed/book_id_to_index.json.
        """
        return load_json(self.path("book_id_to_index.json"))

    @cached_property
    def user_id_to_index(self) -> dict[str, int]:
        """UCSD user_id string → row index in the bulk user matrix.

        Returns:
            Mapping loaded from data/transformed/user_id_to_index.json.
        """
        return load_json(self.path("user_id_to_index.json"))

    @cached_property
    def index_to_book_id(self) -> dict[int, str]:
        """Reverse of `book_id_to_index`, computed once on first access.

        Returns:
            Mapping from row index back to book_id.
        """
        return {v: k for k, v in self.book_id_to_index.items()}

    @cached_property
    def author_embeddings(self) -> np.ndarray:
        """Trained per-author embedding matrix.

        Returns:
            Float32 array of shape (n_authors, embedding_dim).
        """
        return np.load(self.path("author_embeddings.npy"))

    @cached_property
    def author_id_to_index(self) -> dict[str, int]:
        """UCSD author_id string → row index in `author_embeddings`.

        Returns:
            Mapping loaded from data/transformed/author_id_to_index.json.
        """
        return load_json(self.path("author_id_to_index.json"))

    @cached_property
    def book_to_author_idx(self) -> np.ndarray:
        """Per-book primary author row index (or -1 if no author known).

        Returns:
            Int64 array of shape (n_books,).
        """
        return np.load(self.path("book_to_author_idx.npy"))

    @cached_property
    def isbn13_to_book_id(self) -> dict[str, str]:
        """ISBN-13 → UCSD book_id, used by the v4 author fallback in ingest.

        Returns:
            Mapping loaded from data/transformed/isbn13_to_book_id.json.
        """
        return load_json(self.path("isbn13_to_book_id.json"))

    @cached_property
    def book_embeddings(self) -> np.ndarray:
        """Per-book description embedding matrix (v1 feature input).

        Returns:
            Float32 array of shape (n_books, embedding_dim).
        """
        return np.load(self.path("book_embeddings.npy"))

    @cached_property
    def genre_matrix(self) -> np.ndarray:
        """Per-book L2-normalised genre vectors.

        Returns:
            Float32 array of shape (n_books, n_genres).
        """
        return np.load(self.path("genre_matrix.npy"))

    @cached_property
    def num_pages_normalized(self) -> np.ndarray:
        """Per-book normalised page count in [0, 1].

        Returns:
            Float32 array of shape (n_books,).
        """
        return np.load(self.path("num_pages_normalized.npy"))

    @cached_property
    def num_pages_norm_params(self) -> dict[str, float]:
        """{p1, p99, median} parameters used to normalise pages at training time.

        Returns:
            Mapping loaded from data/transformed/num_pages_norm_params.json.
        """
        return load_json(self.path("num_pages_norm_params.json"))

    @cached_property
    def books_meta(self) -> pl.DataFrame:
        """Per-book metadata used at serving for filters and response rendering.

        Returns:
            Polars DataFrame with `book_id`, `title`, `num_pages`, `average_rating`,
            `is_ebook`, and the `genres` struct (one int field per vocab entry).
        """
        return pl.read_parquet(self.path("books_with_genres.parquet"), columns=list(BOOKS_META_COLUMNS))

    def save_book_id_to_index(self, mapping: dict[str, int]) -> None:
        """Persist an updated book_id → index mapping AND invalidate the cache.

        Args:
            mapping: New mapping to write (typically the existing one plus newly
                ingested book_ids).
        """
        with self.path("book_id_to_index.json").open("w", encoding="utf-8") as f:
            json.dump(mapping, f)
        # Drop both the forward and reverse cached values so the next read sees disk.
        self.__dict__.pop("book_id_to_index", None)
        self.__dict__.pop("index_to_book_id", None)


def load_json(path: Path) -> dict:
    """Load a JSON file from disk into a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
