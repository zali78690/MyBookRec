"""Author-embedding lookup for v4 gold features on ingested books.

For each new book the gold pipeline picks an author embedding using a three-tier
priority, designed to maximise quality when we have it and degrade gracefully when
we don't:

1. **Trained**: if the book's ISBN-13 matches a UCSD catalog book, borrow that
   book's trained author embedding directly. Highest quality — author signal
   came from the same model space.
2. **Warm cold-start**: if the new book's first-author name appears in two or
   more rows of the current ingest batch, average those books' description
   embeddings. This is the same definition as training (mean over the author's
   books), just over the ingested subset.
3. **Cold cold-start**: fall back to the book's own description embedding.
   Same shape, no author-specific signal. Better than zeros, worse than (1)
   and (2).

The provenance tier is returned alongside the embedding so callers can log
hit rates per source.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from mybookrec.ingest.schemas import SilverBook
from mybookrec.io.artifacts import TransformedArtifacts
from mybookrec.settings import get_settings


class AuthorEmbeddingLookup:
    """Stateful lookup over the trained author space + an ingest-batch fallback table.

    Build once per ingest pass; query once per book. The constructor holds references
    to the trained matrices; ingest-batch context (`silver_books` + `descriptions`)
    feeds the per-author mean fallback.
    """

    def __init__(
        self,
        isbn13_to_book_id: dict[str, str],
        book_id_to_index: dict[str, int],
        book_to_author_idx: np.ndarray,
        author_embeddings: np.ndarray,
    ) -> None:
        """Wire the trained lookup tables.

        Args:
            isbn13_to_book_id: From `build_isbn_index`.
            book_id_to_index: UCSD book_id → row index.
            book_to_author_idx: Per-book author index (or -1 for missing).
            author_embeddings: (n_authors, dim) trained author embedding matrix.
        """
        self.isbn13_to_book_id = isbn13_to_book_id
        self.book_id_to_index = book_id_to_index
        self.book_to_author_idx = book_to_author_idx
        self.author_embeddings = author_embeddings.astype(np.float32)
        self.embedding_dim = author_embeddings.shape[1]

    def trained_embedding_for(self, isbn_13: str | None) -> np.ndarray | None:
        """Return the trained author embedding for a book matched by ISBN-13.

        Args:
            isbn_13: 13-digit ISBN of the candidate book.

        Returns:
            (dim,) float32 embedding, or None if there's no ISBN match / no author.
        """
        if not isbn_13:
            return None
        book_id = self.isbn13_to_book_id.get(isbn_13)
        if book_id is None:
            return None
        row = self.book_id_to_index.get(book_id)
        if row is None:
            return None
        author_idx = int(self.book_to_author_idx[row])
        if author_idx < 0:
            return None
        return self.author_embeddings[author_idx]

    def resolve(
        self,
        silver: SilverBook,
        own_description_embedding: np.ndarray,
        batch_means: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, str]:
        """Pick the best available author embedding for one book.

        Args:
            silver: The book.
            own_description_embedding: This book's description embedding (fallback).
            batch_means: Output of `compute_batch_author_means` for this ingest pass.

        Returns:
            Tuple of ((dim,) float32 embedding, provenance tier in
            {"trained", "batch_mean", "self"}).
        """
        trained = self.trained_embedding_for(silver.isbn_13)
        if trained is not None:
            return trained, "trained"
        key = author_key(silver)
        if key and key in batch_means:
            return batch_means[key], "batch_mean"
        return own_description_embedding.astype(np.float32), "self"


def author_key(silver: SilverBook) -> str | None:
    """Canonical key for grouping ingested books by first-author name.

    Args:
        silver: A SilverBook.

    Returns:
        Lowercased stripped first-author name, or None if no authors.
    """
    if not silver.authors:
        return None
    return silver.authors[0].strip().lower()


def compute_batch_author_means(
    silver_books: Sequence[SilverBook],
    description_embeddings: np.ndarray,
    min_books_per_author: int = 2,
) -> dict[str, np.ndarray]:
    """Group ingested books by first-author name and average their description embeddings.

    Args:
        silver_books: All silver books in this ingest pass (row-aligned with embeddings).
        description_embeddings: (n_books, dim) embedding matrix.
        min_books_per_author: Minimum books required to include an author. Default 2 means
            singletons fall through to the self-fallback at resolve time.

    Returns:
        Dict mapping author_key → mean embedding. Authors below the threshold are excluded.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, book in enumerate(silver_books):
        key = author_key(book)
        if key:
            groups[key].append(i)
    return {
        key: description_embeddings[rows].mean(axis=0).astype(np.float32)
        for key, rows in groups.items()
        if len(rows) >= min_books_per_author
    }


def load_default_lookup(artifacts: TransformedArtifacts | None = None) -> AuthorEmbeddingLookup:
    """Build the lookup from the standard `data/transformed/` artifacts.

    Args:
        artifacts: Source of the trained-author artifacts. Defaults to one bound at
            settings.transformed_dir.

    Returns:
        Ready-to-query AuthorEmbeddingLookup.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    if artifacts is None:
        artifacts = TransformedArtifacts(get_settings().transformed_dir)
    required = (
        "isbn13_to_book_id.json",
        "book_id_to_index.json",
        "book_to_author_idx.npy",
        "author_embeddings.npy",
    )
    missing = [name for name in required if not artifacts.path(name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {artifacts.base_dir}. Run `python -m mybookrec.features.build_isbn_index` "
            f"and/or `python -m mybookrec.features.build_author_features` first."
        )
    return AuthorEmbeddingLookup(
        isbn13_to_book_id=artifacts.isbn13_to_book_id,
        book_id_to_index=artifacts.book_id_to_index,
        book_to_author_idx=artifacts.book_to_author_idx,
        author_embeddings=artifacts.author_embeddings,
    )
