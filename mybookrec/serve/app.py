"""FastAPI application for real-time book recommendations.

The model, FAISS index, and lookup tables are loaded once at startup via the
lifespan context manager and stashed on `app.state`. Requests run through:

    compute user features  →  encode_user  →  FAISS top-K*oversample
    →  apply post-rank filters  →  return top-K

Behaviour mirrors `mybookrec.recommend`. All configuration comes from
`mybookrec.settings.get_settings()` — no direct env access.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# macOS libomp conflict between FAISS and PyTorch — documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json

import numpy as np
import polars as pl
import torch
from fastapi import FastAPI, HTTPException

from mybookrec.index import build_index, encode_all_items, load_index, query
from mybookrec.io import load_checkpoint, load_item_features
from mybookrec.serve.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from mybookrec.serve.user_features import compute_user_features_from_ratings
from mybookrec.settings import get_settings


def load_book_id_to_index(transformed_dir) -> dict[str, int]:
    """Load the UCSD book_id → row index mapping from disk.

    Args:
        transformed_dir: Path to the data/transformed directory.

    Returns:
        Dict mapping string book_id to integer row index.
    """
    with open(transformed_dir / "book_id_to_index.json") as f:
        return json.load(f)


def flatten_genres_struct(genres_struct: dict | None) -> list[str]:
    """Convert a `{category: count}` struct into the list of categories with a non-zero count.

    Sorted by count descending so the most-prominent genres appear first in responses.

    Args:
        genres_struct: Polars struct field as a dict, or None.

    Returns:
        List of category names present in this book (empty if all counts are null/zero).
    """
    if not isinstance(genres_struct, dict):
        return []
    present = [(name, count or 0) for name, count in genres_struct.items() if count]
    present.sort(key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in present]


def load_meta_dict(transformed_dir) -> dict[str, dict]:
    """Load the per-book metadata used for post-rank filtering and response rendering.

    The on-disk `genres` column is a struct of `{category: count}`. We flatten it here so
    response rendering doesn't need to know the on-disk schema.

    Args:
        transformed_dir: Path to the data/transformed directory.

    Returns:
        Dict mapping book_id → row dict with title, num_pages, average_rating,
        is_ebook, genres (list[str]).
    """
    df = pl.read_parquet(
        transformed_dir / "books_with_genres.parquet",
        columns=["book_id", "title", "num_pages", "average_rating", "is_ebook", "genres"],
    )
    meta: dict[str, dict] = {}
    for row in df.to_dicts():
        row["genres"] = flatten_genres_struct(row.get("genres"))
        meta[str(row["book_id"])] = row
    return meta


def build_or_load_index(model, config, device, index_path):
    """Load the FAISS index from disk if present, otherwise build it on the fly.

    Args:
        model: Loaded TwoTowerModel.
        config: Checkpoint config dict (used to load matching item features).
        device: Torch device string.
        index_path: Optional path to a pre-built .faiss file.

    Returns:
        A populated faiss.Index.
    """
    if index_path is not None and index_path.exists():
        return load_index(index_path)
    item_features = load_item_features(config, device)
    item_embeddings = encode_all_items(model, item_features)
    return build_index(item_embeddings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model, FAISS index, and lookup tables once at startup.

    Args:
        app: The FastAPI app instance; state is stashed on `app.state`.

    Raises:
        FileNotFoundError: If the configured model checkpoint does not exist.
    """
    settings = get_settings()
    model_path = settings.resolved_serve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Serve model checkpoint not found: {model_path}. "
            f"Set MYBOOKREC_SERVE_MODEL_PATH or place a file at {model_path}."
        )

    model, config, _ = load_checkpoint(model_path)
    device = next(model.parameters()).device.type

    transformed_dir = settings.transformed_dir
    index = build_or_load_index(model, config, device, settings.serve_index_path)
    book_id_to_index = load_book_id_to_index(transformed_dir)
    index_to_book_id = {v: k for k, v in book_id_to_index.items()}
    meta_dict = load_meta_dict(transformed_dir)
    book_embeddings = np.load(transformed_dir / "book_embeddings.npy")
    genre_matrix = np.load(transformed_dir / "genre_matrix.npy")
    pages_vec = np.load(transformed_dir / "num_pages_normalized.npy")

    app.state.model = model
    app.state.device = device
    app.state.index = index
    app.state.book_id_to_index = book_id_to_index
    app.state.index_to_book_id = index_to_book_id
    app.state.meta_dict = meta_dict
    app.state.book_embeddings = book_embeddings
    app.state.genre_matrix = genre_matrix
    app.state.pages_vec = pages_vec
    app.state.model_version = model_path.name
    yield


app = FastAPI(title="MyBookRec Serving API", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness + readiness probe.

    Returns:
        HealthResponse with model status and index size.
    """
    model_loaded = getattr(app.state, "model", None) is not None
    index = getattr(app.state, "index", None)
    return HealthResponse(
        status="ok" if model_loaded else "loading",
        model_loaded=model_loaded,
        n_items_in_index=index.ntotal if index is not None else 0,
        model_version=getattr(app.state, "model_version", ""),
    )


def filter_candidates(
    candidate_ids: list[str],
    scores: list[float],
    meta_dict: dict[str, dict],
    excluded: set[str],
    min_avg_rating: float,
    ebook_only: bool,
) -> list[tuple[str, float]]:
    """Apply post-rank filters to FAISS candidates.

    Args:
        candidate_ids: Book ids ranked by similarity (highest first).
        scores: Similarity scores aligned with candidate_ids.
        meta_dict: book_id → metadata row.
        excluded: Book ids to drop unconditionally (already-rated set).
        min_avg_rating: Drop books below this average rating.
        ebook_only: If True, drop non-ebook results.

    Returns:
        Filtered (book_id, score) pairs preserving rank order.
    """
    out: list[tuple[str, float]] = []
    for bid, score in zip(candidate_ids, scores):
        if bid in excluded:
            continue
        row = meta_dict.get(bid, {})
        if (row.get("average_rating") or 0.0) < min_avg_rating:
            continue
        if ebook_only and not (row.get("is_ebook") or False):
            continue
        out.append((bid, float(score)))
    return out


def render_items(
    pairs: list[tuple[str, float]],
    meta_dict: dict[str, dict],
) -> list[RecommendationItem]:
    """Convert (book_id, score) pairs into RecommendationItem response objects.

    Args:
        pairs: Filtered candidates, top-K already applied.
        meta_dict: book_id → metadata row.

    Returns:
        List of RecommendationItem in input order.
    """
    items: list[RecommendationItem] = []
    for bid, score in pairs:
        row = meta_dict.get(bid, {})
        items.append(
            RecommendationItem(
                book_id=bid,
                title=row.get("title") or "",
                score=score,
                average_rating=row.get("average_rating"),
                num_pages=row.get("num_pages"),
                is_ebook=row.get("is_ebook"),
                genres=list(row.get("genres") or []),
            )
        )
    return items


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Compute top-K recommendations for the supplied rating history.

    Args:
        request: Validated request body with ratings, top_k, and filter overrides.

    Returns:
        RecommendResponse with the ranked items, model version, and latency.

    Raises:
        HTTPException: 503 if the model isn't ready, 400 if no ratings matched
            the catalog (so we can't build a user vector).
    """
    settings = get_settings()
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    user_vec = compute_user_features_from_ratings(
        request.ratings,
        app.state.book_id_to_index,
        app.state.book_embeddings,
        app.state.genre_matrix,
        app.state.pages_vec,
    )
    if not np.any(user_vec):
        raise HTTPException(
            status_code=400,
            detail="None of the supplied book_ids matched the catalog — cannot build user vector.",
        )

    device = app.state.device
    user_tensor = torch.from_numpy(user_vec).to(device).float()
    with torch.no_grad():
        user_emb = app.state.model.encode_user(user_tensor).squeeze(0)

    n_candidates = request.top_k * settings.serve_oversample
    scores_arr, indices_arr = query(app.state.index, user_emb, k=n_candidates)
    scores = scores_arr[0].tolist()
    indices = indices_arr[0].tolist()
    index_to_book_id = app.state.index_to_book_id
    candidate_ids = [index_to_book_id[i] for i in indices if i in index_to_book_id]

    min_avg = request.min_avg_rating if request.min_avg_rating is not None else settings.serve_min_avg_rating
    excluded = {r.book_id for r in request.ratings}
    filtered = filter_candidates(
        candidate_ids,
        scores,
        app.state.meta_dict,
        excluded,
        min_avg,
        request.ebook_only,
    )[: request.top_k]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return RecommendResponse(
        recommendations=render_items(filtered, app.state.meta_dict),
        model_version=app.state.model_version,
        latency_ms=latency_ms,
    )
