"""Tests for the silver→gold feature builder (no real embedding model needed)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mybookrec.ingest import to_gold


def test_map_silver_genres_to_vocab_matches_known_keywords() -> None:
    """Expected use: silver "Fantasy" matches the "fantasy, paranormal" vocab entry."""
    vocab = ["fantasy, paranormal", "fiction", "young-adult"]
    vec = to_gold.map_silver_genres_to_vocab(["Fantasy", "Magic", "Fiction"], vocab)
    assert vec[0] > 0  # fantasy keyword hit
    assert vec[1] > 0  # fiction keyword hit
    assert vec[2] == 0  # young-adult absent
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_map_silver_genres_returns_zero_vector_when_no_match() -> None:
    """Edge case: nothing matches → all-zero (no NaN)."""
    vocab = ["fantasy, paranormal", "fiction"]
    vec = to_gold.map_silver_genres_to_vocab(["unmappable thing"], vocab)
    assert np.array_equal(vec, np.zeros(2, dtype=np.float32))


def test_normalize_pages_one_clips_to_p1_p99() -> None:
    """Expected use: pages within [p1, p99] are linearly rescaled to [0, 1]."""
    params = {"p1": 100.0, "p99": 500.0, "median": 300.0}
    assert to_gold.normalize_pages_one(300, params) == pytest.approx(0.5)
    assert to_gold.normalize_pages_one(100, params) == pytest.approx(0.0)
    assert to_gold.normalize_pages_one(500, params) == pytest.approx(1.0)


def test_normalize_pages_one_imputes_missing_with_median() -> None:
    """Failure case: None or 0 pages → median imputation."""
    params = {"p1": 100.0, "p99": 500.0, "median": 300.0}
    assert to_gold.normalize_pages_one(None, params) == pytest.approx(0.5)
    assert to_gold.normalize_pages_one(0, params) == pytest.approx(0.5)


def test_build_features_concatenates_in_right_order() -> None:
    """Expected use: result is [embedding | genre | pages] with correct dims."""
    silver = pl.DataFrame(
        {
            "genres": [["Fantasy"], ["Fiction"]],
            "num_pages": [300, None],
        }
    )
    vocab = ["fantasy, paranormal", "fiction"]
    params = {"p1": 100.0, "p99": 500.0, "median": 300.0}
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    features = to_gold.build_features(silver, vocab, params, embeddings)
    # 3 (embed) + 2 (genre) + 1 (pages) = 6
    assert features.shape == (2, 6)
    # Row 0: embedding [1,0,0] + genre [norm(1,0)=1, 0] + pages [(300-100)/400=0.5]
    assert features[0, 0] == 1.0
    assert features[0, 3] == 1.0
    assert features[0, 5] == pytest.approx(0.5)
