"""Shared IO helpers for the CLI scripts.

The split between this module and `mybookrec.features.*` is deliberate:
this is a thin runtime IO layer the CLIs share at inference/training time
(loading checkpoints, feature matrices, id mappings). The `features` package
contains the offline data-pipeline code that *produces* those artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch

from mybookrec import DATA_DIR
from mybookrec.model.towers import TwoTowerModel


# Feature-set registry — single source of truth for "given a checkpoint's
# input dims, which on-disk files match?". Adding a new feature set means
# adding one entry here instead of touching every loader.
class FeatureSet:
    """One named feature set: dims + the on-disk filenames they live in."""

    def __init__(
        self,
        name: str,
        user_input_dim: int,
        item_input_dim: int,
        bulk_user_npy: str,
        personal_user_npy: str,
        item_files: tuple[str, ...],
    ) -> None:
        """Register a feature set.

        Args:
            name: Human-readable identifier (e.g. "v1", "v4").
            user_input_dim: Dimensionality of the user feature vector.
            item_input_dim: Dimensionality of the item feature vector.
            bulk_user_npy: Filename under data/transformed/ for the
                full user matrix used during training.
            personal_user_npy: Filename for the single-row personal user vector.
            item_files: One or more filenames whose concatenation produces
                the item feature matrix.
        """
        self.name = name
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.bulk_user_npy = bulk_user_npy
        self.personal_user_npy = personal_user_npy
        self.item_files = item_files


FEATURE_SETS: tuple[FeatureSet, ...] = (
    FeatureSet(
        name="v1",
        user_input_dim=779,
        item_input_dim=395,
        bulk_user_npy="train_user_features.npy",
        personal_user_npy="user_features.npy",
        item_files=("book_embeddings.npy", "genre_matrix.npy", "num_pages_normalized.npy"),
    ),
    FeatureSet(
        name="v4",
        user_input_dim=1163,
        item_input_dim=779,
        bulk_user_npy="train_user_features_v4.npy",
        personal_user_npy="user_features_v4.npy",
        item_files=("item_features_v4.npy",),
    ),
)


def select_device() -> str:
    """Pick the best torch device available.

    Returns:
        "mps" on Apple Silicon, "cuda" if a GPU is available, else "cpu".
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_feature_set(item_input_dim: int) -> FeatureSet:
    """Look up the FeatureSet matching an item feature dimensionality.

    Args:
        item_input_dim: The ItemTower's input dim (read from a checkpoint config).

    Returns:
        The matching FeatureSet.

    Raises:
        ValueError: If no registered feature set matches.
    """
    for fs in FEATURE_SETS:
        if fs.item_input_dim == item_input_dim:
            return fs
    known = ", ".join(f"{fs.name}={fs.item_input_dim}" for fs in FEATURE_SETS)
    raise ValueError(f"No FeatureSet for item_input_dim={item_input_dim} (known: {known})")


def detect_available_feature_set() -> FeatureSet:
    """Probe disk for the highest-numbered feature set whose files are present.

    Returns:
        The latest FeatureSet whose item files exist, falling back to the first
        registered set if nothing newer is present.
    """
    transformed = DATA_DIR / "transformed"
    for fs in reversed(FEATURE_SETS):
        if all((transformed / f).exists() for f in fs.item_files):
            return fs
    return FEATURE_SETS[0]


def load_npy_to_device(path: Path, device: str) -> torch.Tensor:
    """Load a .npy file as a float32 torch tensor on the given device.

    Args:
        path: Path to the .npy file.
        device: Target torch device string ("cpu", "cuda", "mps").

    Returns:
        Float32 tensor on the requested device.
    """
    return torch.from_numpy(np.load(path)).to(device).float()


def load_item_features(config: dict, device: str) -> torch.Tensor:
    """Load the item feature matrix for the feature set keyed by the checkpoint.

    Args:
        config: Checkpoint config dict, must contain "item_input_dim".
        device: Target torch device.

    Returns:
        Item feature tensor of shape (n_items, item_input_dim).
    """
    fs = resolve_feature_set(config["item_input_dim"])
    transformed = DATA_DIR / "transformed"
    tensors = [load_npy_to_device(transformed / f, device) for f in fs.item_files]
    if len(tensors) == 1:
        return tensors[0]
    # Multi-file v1-style: book_embeddings + genre_matrix + num_pages_normalized.
    # Last file is a 1-D pages vector and needs reshaping.
    tensors[-1] = tensors[-1].reshape(-1, 1)
    return torch.cat(tensors, dim=1).contiguous()


def load_train_user_features(config: dict, device: str) -> torch.Tensor:
    """Load the bulk train-user feature matrix.

    Args:
        config: Checkpoint config dict, must contain "item_input_dim"
            (used to resolve the matching FeatureSet).
        device: Target torch device.

    Returns:
        Bulk user feature tensor of shape (n_users, user_input_dim).
    """
    fs = resolve_feature_set(config["item_input_dim"])
    return load_npy_to_device(DATA_DIR / "transformed" / fs.bulk_user_npy, device)


def personal_user_features_path(config: dict) -> Path:
    """Resolve the on-disk path for the single-row personal user vector.

    Args:
        config: Checkpoint config dict.

    Returns:
        Path to the personal user .npy file matching the checkpoint's feature set.
    """
    fs = resolve_feature_set(config["item_input_dim"])
    return DATA_DIR / "transformed" / fs.personal_user_npy


def load_features_for_checkpoint(config: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the bulk user features and item features for a checkpoint.

    Args:
        config: Checkpoint config dict.
        device: Target torch device.

    Returns:
        Tuple of (bulk_user_features, item_features) tensors.
    """
    return load_train_user_features(config, device), load_item_features(config, device)


def load_checkpoint(
    checkpoint_path: Path,
    device: str | None = None,
) -> tuple[TwoTowerModel, dict, dict]:
    """Load a trained TwoTowerModel from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt file written by `save_checkpoint`.
        device: Target device. If None, picks the best available via `select_device`.

    Returns:
        Tuple of (model in eval mode, config dict, full checkpoint dict).
    """
    if device is None:
        device = select_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = TwoTowerModel(
        user_input_dim=config["user_input_dim"],
        item_input_dim=config["item_input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config, ckpt


def batch_encode(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Run an encoder over features in batches and concatenate the outputs.

    Used identically by FAISS index building, evaluation, vibe checks, and the
    in-training evaluator — same loop, one implementation.

    Args:
        encoder: Callable mapping a feature slice to an embedding tensor.
            Typically `model.encode_item` or `model.encode_user`.
        features: Input features of shape (n, feature_dim).
        batch_size: Number of rows per forward pass.

    Returns:
        Concatenated embeddings of shape (n, embedding_dim).
    """
    chunks = [encoder(features[i : i + batch_size]) for i in range(0, features.shape[0], batch_size)]
    return torch.cat(chunks, dim=0)


def load_id_mappings() -> tuple[dict[str, int], dict[str, int]]:
    """Load the user-id and book-id → integer-index JSON mappings.

    Returns:
        Tuple of (user_id_to_index, book_id_to_index).
    """
    transformed = DATA_DIR / "transformed"
    with open(transformed / "user_id_to_index.json") as f:
        user_id_to_index = json.load(f)
    with open(transformed / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)
    return user_id_to_index, book_id_to_index


def id_map_df(mapping: dict[str, int], id_col: str, idx_col: str) -> pl.DataFrame:
    """Materialize an id-to-index dict as a typed polars DataFrame for join use.

    Args:
        mapping: dict of {string id: integer index}.
        id_col: Name of the id column.
        idx_col: Name of the index column.

    Returns:
        Polars DataFrame with one row per mapping entry.
    """
    return pl.DataFrame(
        {id_col: list(mapping.keys()), idx_col: list(mapping.values())},
        schema={id_col: pl.String, idx_col: pl.Int64},
    )


def load_split_interactions(
    data_split: str,
    user_id_to_index: dict[str, int] | None = None,
    book_id_to_index: dict[str, int] | None = None,
    min_rating: int | None = None,
) -> pl.DataFrame:
    """Load a slice of training_interactions joined onto integer indices.

    Args:
        data_split: One of "train", "validation", "test".
        user_id_to_index: Optional pre-loaded user mapping. Loaded from disk if None.
        book_id_to_index: Optional pre-loaded book mapping. Loaded from disk if None.
        min_rating: If set, keep only rows with rating >= this value.

    Returns:
        Polars DataFrame with columns user_id, book_id, rating, user_idx, book_idx.
    """
    if user_id_to_index is None or book_id_to_index is None:
        user_id_to_index, book_id_to_index = load_id_mappings()

    user_map = id_map_df(user_id_to_index, "user_id", "user_idx")
    book_map = id_map_df(book_id_to_index, "book_id", "book_idx")

    q = (
        pl.scan_parquet(DATA_DIR / "transformed" / "training_interactions.parquet")
        .filter(pl.col("data_split") == data_split)
        .select("user_id", "book_id", "rating")
        .with_columns(pl.col("book_id").cast(pl.String))
        .join(user_map.lazy(), on="user_id", how="left")
        .join(book_map.lazy(), on="book_id", how="left")
        .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
    )
    if min_rating is not None:
        q = q.filter(pl.col("rating") >= min_rating)
    return q.collect()


def build_train_exclude() -> dict[int, np.ndarray]:
    """Map each user to the set of book indices they rated in the train split.

    Used at eval time to mask already-rated books from candidate recommendations.

    Returns:
        Dict from user_idx to a numpy int64 array of book indices.
    """
    train_df = load_split_interactions("train")
    grouped = train_df.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
    return {row["user_idx"]: np.array(row["rated_books"], dtype=np.int64) for row in grouped.to_dicts()}


def sample_test_pairs(n_pairs: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample held-out test (user_idx, positive_book_idx) pairs.

    Args:
        n_pairs: How many pairs to sample (or all if fewer exist).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (user_idxs, book_idxs) numpy arrays.
    """
    test_df = load_split_interactions("test", min_rating=4)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_df), size=min(n_pairs, len(test_df)), replace=False)
    sample = test_df[indices]
    return sample["user_idx"].to_numpy(), sample["book_idx"].to_numpy()
