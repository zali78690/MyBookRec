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

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mybookrec.ingest.schemas import SilverBook
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

    @staticmethod
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

    @classmethod
    def build_batch_means(
        cls,
        silver_books: Sequence[SilverBook],
        description_embeddings: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Group ingested books by first-author name and average their description embeddings.

        Args:
            silver_books: All silver books in this ingest pass (row-aligned with embeddings).
            description_embeddings: (n_books, dim) embedding matrix.

        Returns:
            Dict mapping author_key → mean embedding. Authors with only one book are
            still included; callers can decide whether to use the single-book mean.
        """
        groups: dict[str, list[int]] = defaultdict(list)
        for i, book in enumerate(silver_books):
            key = cls.author_key(book)
            if key:
                groups[key].append(i)
        return {
            key: description_embeddings[rows].mean(axis=0).astype(np.float32)
            for key, rows in groups.items()
            if len(rows) >= 2
        }

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
            batch_means: Output of `build_batch_means` for this ingest pass.

        Returns:
            Tuple of ((dim,) float32 embedding, provenance tier in
            {"trained", "batch_mean", "self"}).
        """
        trained = self.trained_embedding_for(silver.isbn_13)
        if trained is not None:
            return trained, "trained"
        key = self.author_key(silver)
        if key and key in batch_means:
            return batch_means[key], "batch_mean"
        return own_description_embedding.astype(np.float32), "self"


def load_default_lookup() -> AuthorEmbeddingLookup:
    """Build the lookup from the standard `data/transformed/` artifacts.

    Returns:
        Ready-to-query AuthorEmbeddingLookup.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    settings = get_settings()
    transformed = settings.transformed_dir
    required = [
        "isbn13_to_book_id.json",
        "book_id_to_index.json",
        "book_to_author_idx.npy",
        "author_embeddings.npy",
    ]
    for filename in required:
        if not (transformed / filename).exists():
            raise FileNotFoundError(
                f"Missing {filename} in {transformed}. Run `python -m mybookrec.features.build_isbn_index` "
                f"(for isbn13_to_book_id) or `python -m mybookrec.features.build_author_features` "
                f"(for the others) first."
            )
    return AuthorEmbeddingLookup(
        isbn13_to_book_id=read_json_file(transformed / "isbn13_to_book_id.json"),
        book_id_to_index=read_json_file(transformed / "book_id_to_index.json"),
        book_to_author_idx=np.load(transformed / "book_to_author_idx.npy"),
        author_embeddings=np.load(transformed / "author_embeddings.npy"),
    )


def read_json_file(path: Path) -> dict:
    """Read a JSON file from disk into a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
