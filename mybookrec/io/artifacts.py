"""Typed lazy loaders for everything under data/transformed/.

One module owns the on-disk filenames + deserialisation conventions for the
artifacts the training pipeline produces. Callers (serve, recommend, ingest,
eval) stop knowing filenames; they ask the registry for a named artifact.

Layout the loaders point at (see plans/book-recommender-mvp-plan.md):

  data/transformed/
  ├── shared/        # model-independent: id mappings, books_with_genres,
  │                  # training_interactions, genre matrix, page norms,
  │                  # author lookups, isbn index, my_books, embedding_input
  └── v1_minilm/     # MiniLM-384 embeddings + downstream features
                     # (future v2_mxbai/ when the mxbai pass completes)

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


DEFAULT_MODEL_RUN = "v1_minilm"
SHARED_SUBDIR = "shared"


class TransformedArtifacts:
    """Lazy loader for every file the training pipeline writes to data/transformed/.

    Constructor is cheap; properties load on first access and cache thereafter.
    Inject this into modules that need artifacts so they don't need to know
    filenames or how each artifact deserialises.

    Two subdirs:
      shared/      — files independent of the embedding model (id mappings,
                     genre matrix, page norms, books_with_genres, ...).
      <model_run>/ — embedding-model-specific artifacts (book_embeddings,
                     author_embeddings, item_features, user_features, ...).
                     Defaults to "v1_minilm"; pass `model_run="v2_mxbai"` once
                     the mxbai pass completes.
    """

    def __init__(self, base_dir: Path, model_run: str = DEFAULT_MODEL_RUN) -> None:
        """Bind to a transformed-data root + an embedding-model run.

        Args:
            base_dir: Usually `settings.transformed_dir`. Tests pass a tmp_path.
            model_run: Name of the embedding-model run subdirectory (e.g.
                "v1_minilm", "v2_mxbai"). Resolved relative to `base_dir`.
        """
        self.base_dir = Path(base_dir)
        self.model_run = model_run

    def path(self, filename: str) -> Path:
        """Resolve an artifact's path without loading it.

        Filenames containing a `/` are treated as already-qualified (relative
        to `base_dir`) — useful for callers that want to point at a specific
        subdir. Bare filenames resolve under `shared/`.

        Args:
            filename: A bare filename ("book_id_to_index.json") or a path
                relative to base_dir ("v1_minilm/book_embeddings.npy").

        Returns:
            Absolute path under `base_dir`.
        """
        if "/" in filename:
            return self.base_dir / filename
        return self.base_dir / SHARED_SUBDIR / filename

    def model_path(self, filename: str) -> Path:
        """Resolve a path under the current `model_run` subdir.

        Args:
            filename: Bare filename like "book_embeddings.npy".

        Returns:
            Absolute path under `base_dir/<model_run>/`.
        """
        return self.base_dir / self.model_run / filename

    @cached_property
    def book_id_to_index(self) -> dict[str, int]:
        """UCSD book_id string → row index in the item matrices.

        Returns:
            Mapping loaded from shared/book_id_to_index.json.
        """
        return load_json(self.path("book_id_to_index.json"))

    @cached_property
    def user_id_to_index(self) -> dict[str, int]:
        """UCSD user_id string → row index in the bulk user matrix.

        Returns:
            Mapping loaded from shared/user_id_to_index.json.
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
        return np.load(self.model_path("author_embeddings.npy"))

    @cached_property
    def author_id_to_index(self) -> dict[str, int]:
        """UCSD author_id string → row index in `author_embeddings`.

        Returns:
            Mapping loaded from shared/author_id_to_index.json.
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
            Mapping loaded from shared/isbn13_to_book_id.json.
        """
        return load_json(self.path("isbn13_to_book_id.json"))

    @cached_property
    def book_embeddings(self) -> np.ndarray:
        """Per-book description embedding matrix for the active model run.

        Returns:
            Float32 array of shape (n_books, embedding_dim).
        """
        return np.load(self.model_path("book_embeddings.npy"))

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
            Mapping loaded from shared/num_pages_norm_params.json.
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
