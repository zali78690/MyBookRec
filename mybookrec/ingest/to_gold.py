"""Silver → gold: embed descriptions and produce feature vectors for new books.

Outputs (under data/gold/):
- `books.parquet`     — silver schema rows for serving metadata.
- `embeddings.npy`    — description embeddings, same model as training.
- `item_features.npy` — full v1 item feature vector (embed + genre + pages), one row per book.
- `book_ids.json`     — ordered list of book_ids aligned with the matrices.

The v1 schema (`embed_dim + 10 genre + 1 page = 395`) is the only one produced here.
v4 author features need an existing author embedding lookup, which only ingests of known
authors satisfy — deferred until needed.

Genre mapping for new books is substring-based against the existing 10-category vocab:
free-text silver genres are lowercased and matched against vocab keywords. Crude but
deterministic, and good enough for FAISS retrieval where the description embedding
dominates anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from mybookrec import ROOT_DIR
from mybookrec.ingest.author_lookup import AuthorEmbeddingLookup, load_default_lookup
from mybookrec.ingest.schemas import SilverBook
from mybookrec.settings import get_settings


def load_vocab() -> list[str]:
    """Load the 10-category genre vocab from the features package.

    Returns:
        The ordered vocab list.
    """
    with open(ROOT_DIR / "mybookrec" / "features" / "genre_vocab.json") as f:
        return json.load(f)


def load_page_norm_params() -> dict[str, float]:
    """Load the {p1, p99, median} normalisation params for num_pages.

    Returns:
        Dict with float keys p1, p99, median.

    Raises:
        FileNotFoundError: If the params file hasn't been produced yet.
    """
    settings = get_settings()
    path = settings.transformed_dir / "num_pages_norm_params.json"
    with open(path) as f:
        return json.load(f)


def map_silver_genres_to_vocab(silver_genres: list[str], vocab: list[str]) -> np.ndarray:
    """Project free-text silver genres onto the fixed-width Goodreads vocab via keyword match.

    Each vocab entry like "fantasy, paranormal" is split on commas; if any silver genre
    contains any keyword, that column is incremented. The result is L2-normalised.

    Args:
        silver_genres: Free-text genre strings from a silver book row.
        vocab: Ordered vocab list (length 10).

    Returns:
        Float32 vector shape (len(vocab),). All-zero if no silver genre matches.
    """
    counts = np.zeros(len(vocab), dtype=np.float32)
    lowered = [g.lower() for g in silver_genres if isinstance(g, str)]
    for col, keywords_csv in enumerate(vocab):
        keywords = [k.strip().lower() for k in keywords_csv.split(",")]
        for keyword in keywords:
            if any(keyword in g for g in lowered):
                counts[col] += 1
                break
    norm = float(np.linalg.norm(counts))
    return counts / norm if norm > 0 else counts


def normalize_pages_one(num_pages: int | None, params: dict[str, float]) -> float:
    """Normalise a single page count using saved {p1, p99, median} params.

    Args:
        num_pages: Pages for one book, or None / 0 (treated as missing → imputed with median).
        params: Output of `load_page_norm_params`.

    Returns:
        Float in [0, 1].
    """
    p1, p99, median = params["p1"], params["p99"], params["median"]
    value = float(num_pages) if num_pages and num_pages > 0 else median
    clipped = max(p1, min(p99, value))
    if p99 == p1:
        return 0.0
    return (clipped - p1) / (p99 - p1)


def embed_descriptions(titles: list[str], descriptions: list[str | None], model_name: str) -> np.ndarray:
    """Embed `title. description` strings with the configured sentence-transformer.

    Args:
        titles: Title per book (parallel-indexed with descriptions).
        descriptions: Description per book (None replaced by empty string).
        model_name: HF model id.

    Returns:
        Float32 matrix shape (n_books, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    texts = [f"{title}. {description or ''}".strip() for title, description in zip(titles, descriptions)]
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, batch_size=64, show_progress_bar=False)).astype(np.float32)


def build_features(
    silver: pl.DataFrame,
    vocab: list[str],
    page_params: dict[str, float],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Concatenate description embeddings + genre vector + normalised pages (v1 schema).

    Args:
        silver: Silver parquet (one row per book).
        vocab: Genre vocab list.
        page_params: Page normalisation params.
        embeddings: (n_books, embedding_dim) description embeddings, row-aligned with silver.

    Returns:
        Float32 matrix shape (n_books, embedding_dim + len(vocab) + 1).
    """
    genre_matrix = np.stack(
        [map_silver_genres_to_vocab(g, vocab) for g in silver["genres"].to_list()],
        axis=0,
    ).astype(np.float32)
    pages_vec = np.asarray(
        [normalize_pages_one(p, page_params) for p in silver["num_pages"].to_list()],
        dtype=np.float32,
    ).reshape(-1, 1)
    return np.concatenate([embeddings, genre_matrix, pages_vec], axis=1)


def build_author_features(
    silver_books: list[SilverBook],
    description_embeddings: np.ndarray,
    lookup: AuthorEmbeddingLookup,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve a per-book author embedding via the trained / batch-mean / self fallback chain.

    Args:
        silver_books: Row-aligned list of SilverBook records.
        description_embeddings: (n_books, dim) description embeddings.
        lookup: Initialised AuthorEmbeddingLookup.

    Returns:
        Tuple of:
        - (n_books, dim) float32 author-embedding matrix.
        - Provenance counts dict, e.g. {"trained": 12, "batch_mean": 3, "self": 5}.
    """
    batch_means = lookup.build_batch_means(silver_books, description_embeddings)
    out = np.zeros_like(description_embeddings, dtype=np.float32)
    provenance: dict[str, int] = {"trained": 0, "batch_mean": 0, "self": 0}
    for i, book in enumerate(silver_books):
        emb, source = lookup.resolve(book, description_embeddings[i], batch_means)
        out[i] = emb
        provenance[source] += 1
    return out, provenance


def run(
    model_name: str | None = None,
    silver_path: Path | None = None,
    feature_set: str = "v1",
) -> tuple[Path, int]:
    """Read silver parquet → write gold metadata + embeddings + features.

    Args:
        model_name: HF model id; defaults to settings.embed_model_name.
        silver_path: Source silver parquet; defaults to settings.silver_dir / books.parquet.
        feature_set: "v1" (395-dim: emb+genre+pages) or "v4" (779-dim: v1 + author emb).
            v4 requires the trained author artifacts plus `isbn13_to_book_id.json`.

    Returns:
        Tuple of (path to gold/books.parquet, number of books written).
    """
    if feature_set not in ("v1", "v4"):
        raise ValueError(f"feature_set must be 'v1' or 'v4', got {feature_set!r}")

    settings = get_settings()
    if model_name is None:
        model_name = settings.embed_model_name
    if silver_path is None:
        silver_path = settings.silver_dir / "books.parquet"

    silver = pl.read_parquet(silver_path)
    silver_books = [SilverBook(**row) for row in silver.to_dicts()]
    vocab = load_vocab()
    page_params = load_page_norm_params()

    embeddings = embed_descriptions(
        silver["title"].to_list(),
        silver["description"].to_list(),
        model_name,
    )
    item_features = build_features(silver, vocab, page_params, embeddings)

    if feature_set == "v4":
        lookup = load_default_lookup()
        author_features, provenance = build_author_features(silver_books, embeddings, lookup)
        item_features = np.concatenate([item_features, author_features], axis=1).astype(np.float32)
        total = sum(provenance.values()) or 1
        breakdown = ", ".join(f"{src}={n} ({100 * n / total:.0f}%)" for src, n in provenance.items())
        print(f"[gold/v4] author embedding provenance: {breakdown}")

    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    out_books = settings.gold_dir / "books.parquet"
    silver.write_parquet(out_books, compression="zstd")
    np.save(settings.gold_dir / "embeddings.npy", embeddings)
    np.save(settings.gold_dir / "item_features.npy", item_features)
    with open(settings.gold_dir / "book_ids.json", "w") as f:
        json.dump(silver["book_id"].to_list(), f)

    return out_books, silver.shape[0]
