"""Tests for the TransformedArtifacts lazy loader."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import polars as pl
import pytest

from mybookrec.io.artifacts import TransformedArtifacts


@pytest.fixture
def artifact_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tmp dir pre-populated with one fixture per artifact a real pipeline would write.

    Mirrors the on-disk layout: shared files under `shared/`, model-specific files
    under `v1_minilm/` (the default model run).
    """
    shared = tmp_path / "shared"
    minilm = tmp_path / "v1_minilm"
    shared.mkdir()
    minilm.mkdir()
    (shared / "book_id_to_index.json").write_text(json.dumps({"book_a": 0, "book_b": 1}))
    (shared / "user_id_to_index.json").write_text(json.dumps({"user_x": 0, "user_y": 1}))
    (shared / "author_id_to_index.json").write_text(json.dumps({"auth_1": 0}))
    (shared / "isbn13_to_book_id.json").write_text(json.dumps({"9780000000001": "book_a"}))
    (shared / "num_pages_norm_params.json").write_text(json.dumps({"p1": 50.0, "p99": 800.0, "median": 300.0}))
    np.save(minilm / "author_embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    np.save(shared / "book_to_author_idx.npy", np.array([0, -1], dtype=np.int64))
    np.save(minilm / "book_embeddings.npy", np.zeros((2, 4), dtype=np.float32))
    np.save(shared / "genre_matrix.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(shared / "num_pages_normalized.npy", np.array([0.1, 0.5], dtype=np.float32))
    pl.DataFrame(
        {
            "book_id": ["book_a", "book_b"],
            "title": ["A", "B"],
            "num_pages": [200, 400],
            "average_rating": [4.1, 3.5],
            "is_ebook": [True, False],
            "genres": [{"fiction": 5}, {"fiction": 2}],
        }
    ).write_parquet(shared / "books_with_genres.parquet")
    return tmp_path


def test_expected_use_loads_each_artifact_by_name(artifact_dir: pathlib.Path) -> None:
    """Expected use: every named loader returns the correct deserialised value."""
    artifacts = TransformedArtifacts(artifact_dir)
    assert artifacts.book_id_to_index == {"book_a": 0, "book_b": 1}
    assert artifacts.user_id_to_index == {"user_x": 0, "user_y": 1}
    assert artifacts.isbn13_to_book_id == {"9780000000001": "book_a"}
    assert artifacts.num_pages_norm_params["median"] == 300.0
    assert artifacts.author_embeddings.shape == (1, 2)
    assert artifacts.book_to_author_idx.tolist() == [0, -1]
    assert artifacts.books_meta.shape == (2, 6)


def test_index_to_book_id_is_reverse_of_book_id_to_index(artifact_dir: pathlib.Path) -> None:
    """Edge case: the reverse mapping is derived once from the forward one."""
    artifacts = TransformedArtifacts(artifact_dir)
    assert artifacts.index_to_book_id == {0: "book_a", 1: "book_b"}


def test_loaders_are_cached_across_accesses(artifact_dir: pathlib.Path) -> None:
    """Expected use: cached_property means repeated reads return the same object identity."""
    artifacts = TransformedArtifacts(artifact_dir)
    first = artifacts.book_id_to_index
    second = artifacts.book_id_to_index
    assert first is second  # exact same dict object — not just equal


def test_save_book_id_to_index_persists_and_invalidates_cache(artifact_dir: pathlib.Path) -> None:
    """Expected use: save writes to disk AND drops cached forward + reverse maps."""
    artifacts = TransformedArtifacts(artifact_dir)
    original = artifacts.book_id_to_index
    original_reverse = artifacts.index_to_book_id

    new_mapping = {**original, "book_c": 2}
    artifacts.save_book_id_to_index(new_mapping)

    # Disk reflects the new mapping (under shared/).
    assert json.loads((artifact_dir / "shared" / "book_id_to_index.json").read_text()) == new_mapping
    # Cache was invalidated — next read sees the extended mapping.
    assert artifacts.book_id_to_index == new_mapping
    assert artifacts.index_to_book_id == {0: "book_a", 1: "book_b", 2: "book_c"}
    # And it's a new object (cache was indeed dropped, not reused).
    assert artifacts.index_to_book_id is not original_reverse


def test_failure_case_missing_artifact_raises_file_not_found(tmp_path: pathlib.Path) -> None:
    """Failure case: asking for an artifact that doesn't exist raises FileNotFoundError."""
    artifacts = TransformedArtifacts(tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = artifacts.author_embeddings


def test_path_resolves_bare_filenames_under_shared(tmp_path: pathlib.Path) -> None:
    """Edge case: bare filenames go under shared/; slash-paths resolve verbatim."""
    artifacts = TransformedArtifacts(tmp_path)
    assert artifacts.path("anything.json") == tmp_path / "shared" / "anything.json"
    assert artifacts.path("v2_mxbai/anything.npy") == tmp_path / "v2_mxbai" / "anything.npy"


def test_model_path_resolves_under_active_model_run(tmp_path: pathlib.Path) -> None:
    """Edge case: model_path joins under the configured model_run subdir."""
    artifacts = TransformedArtifacts(tmp_path, model_run="v2_mxbai")
    assert artifacts.model_path("book_embeddings.npy") == tmp_path / "v2_mxbai" / "book_embeddings.npy"
