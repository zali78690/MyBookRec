"""Pydantic request and response models for the serving API.

Field types use Python 3.12 union syntax (`str | None`) and live exclusively at
the API boundary; internal modules pass numpy arrays / torch tensors instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mybookrec.settings import get_settings


class RatingInput(BaseModel):
    """One user rating supplied with a /recommend request.

    Attributes:
        book_id: UCSD book identifier as a string (matches `book_id_to_index.json`).
        rating: Integer 1-5 star rating. 3-star is allowed but excluded from
            like/dislike aggregation (kept for parity with the offline pipeline).
    """

    book_id: str = Field(..., description="UCSD book_id (string).")
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating.")


class RecommendRequest(BaseModel):
    """Request body for POST /recommend.

    Attributes:
        ratings: User's rating history (at least one entry required).
        top_k: Number of recommendations to return after filtering.
        min_avg_rating: Optional quality gate; if None, the server uses
            `settings.serve_min_avg_rating`.
        ebook_only: If True, restrict results to `is_ebook=True` books.
    """

    ratings: list[RatingInput] = Field(..., min_length=1, description="User rating history.")
    top_k: int = Field(
        default_factory=lambda: get_settings().serve_default_top_k,
        ge=1,
        le=200,
        description="How many recommendations to return after filtering.",
    )
    min_avg_rating: float | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Override the server's default min average rating filter.",
    )
    ebook_only: bool = Field(default=False, description="Restrict to ebook editions only.")


class RecommendationItem(BaseModel):
    """One recommended book in a /recommend response.

    Attributes:
        book_id: UCSD book identifier.
        title: Book title (may be empty for incomplete catalog rows).
        score: Inner-product similarity from the two-tower model.
        average_rating: Global Goodreads average (None if catalog missing it).
        num_pages: Page count (None if unknown).
        is_ebook: True/False/None tri-state from the catalog.
        genres: List of inferred genre strings (may be empty).
    """

    book_id: str
    title: str
    score: float
    average_rating: float | None = None
    num_pages: int | None = None
    is_ebook: bool | None = None
    genres: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    """Response body for POST /recommend.

    Attributes:
        recommendations: Top-K filtered results, highest score first.
        model_version: Filename stem of the loaded checkpoint.
        latency_ms: End-to-end server-side latency in milliseconds.
    """

    recommendations: list[RecommendationItem]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response body for GET /healthz.

    Attributes:
        status: "ok" when the model + index are loaded.
        model_loaded: Whether the model is in app.state.
        n_items_in_index: Number of vectors in the FAISS index.
        model_version: Filename of the loaded checkpoint.
    """

    status: str
    model_loaded: bool
    n_items_in_index: int
    model_version: str
