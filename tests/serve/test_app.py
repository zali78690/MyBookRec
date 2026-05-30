"""Tests for the FastAPI app's filter + render helpers.

The lifespan that loads the real 1.78M-item index is too heavy for unit tests.
These tests exercise the pure helpers + the request validation surface using
TestClient with a hand-constructed app.state.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mybookrec.serve import app as app_module
from mybookrec.serve.schemas import RecommendRequest


def test_filter_candidates_drops_excluded_low_rating_and_non_ebook() -> None:
    """Expected use: all three filter dimensions are applied in one pass."""
    meta = {
        "a": {"average_rating": 4.5, "is_ebook": True},
        "b": {"average_rating": 3.0, "is_ebook": True},
        "c": {"average_rating": 4.7, "is_ebook": False},
        "d": {"average_rating": 4.9, "is_ebook": True},
    }
    out = app_module.filter_candidates(
        candidate_ids=["a", "b", "c", "d"],
        scores=[0.9, 0.8, 0.7, 0.6],
        meta_dict=meta,
        excluded={"d"},
        min_avg_rating=4.0,
        ebook_only=True,
    )
    # b drops on rating, c drops on ebook, d drops on excluded → only a remains.
    assert out == [("a", pytest.approx(0.9))]


def test_filter_candidates_preserves_rank_order() -> None:
    """Edge case: filtered output keeps input rank order."""
    meta = {bid: {"average_rating": 5.0, "is_ebook": True} for bid in "abcd"}
    out = app_module.filter_candidates(
        candidate_ids=["d", "a", "c", "b"],
        scores=[0.9, 0.8, 0.7, 0.6],
        meta_dict=meta,
        excluded=set(),
        min_avg_rating=0.0,
        ebook_only=False,
    )
    assert [bid for bid, _ in out] == ["d", "a", "c", "b"]


def test_filter_candidates_handles_missing_metadata() -> None:
    """Failure case: candidates with no metadata are dropped (avg_rating defaults to 0)."""
    out = app_module.filter_candidates(
        candidate_ids=["missing"],
        scores=[0.99],
        meta_dict={},
        excluded=set(),
        min_avg_rating=4.0,
        ebook_only=False,
    )
    assert out == []


def test_flatten_genres_struct_returns_present_categories_sorted_by_count() -> None:
    """Expected use: most-prominent categories appear first; absent ones omitted."""
    struct = {
        "fantasy": 100,
        "romance": None,
        "young-adult": 50,
        "fiction": 0,
        "comics": 200,
    }
    assert app_module.flatten_genres_struct(struct) == ["comics", "fantasy", "young-adult"]


def test_flatten_genres_struct_handles_none() -> None:
    """Failure case: None / non-dict input → empty list."""
    assert app_module.flatten_genres_struct(None) == []
    assert app_module.flatten_genres_struct("garbage") == []  # type: ignore[arg-type]


def test_recommend_request_rejects_empty_ratings() -> None:
    """Failure case: at least one rating is required."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        RecommendRequest(ratings=[])


def test_recommend_endpoint_returns_400_when_model_unloaded() -> None:
    """Failure case: when model isn't loaded, 503 is raised before any work."""
    # Skip the lifespan and ensure state is empty.
    client = TestClient(app_module.app, raise_server_exceptions=False)
    # No lifespan startup → app.state has no `model` attribute.
    response = client.post(
        "/recommend",
        json={"ratings": [{"book_id": "x", "rating": 5}], "top_k": 3},
    )
    assert response.status_code == 503


def test_healthz_responds_even_without_model() -> None:
    """Edge case: /healthz works pre-startup (reports loading state)."""
    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "loading"}
    assert "model_loaded" in body


def test_render_items_produces_recommendation_payload() -> None:
    """Expected use: filtered pairs become RecommendationItem objects with metadata."""
    meta = {
        "a": {
            "title": "The Final Empire",
            "average_rating": 4.5,
            "num_pages": 541,
            "is_ebook": True,
            "genres": ["fantasy"],
        }
    }
    items = app_module.render_items([("a", 0.9)], meta)
    assert len(items) == 1
    assert items[0].book_id == "a"
    assert items[0].title == "The Final Empire"
    assert np.isclose(items[0].score, 0.9)
    assert items[0].genres == ["fantasy"]
