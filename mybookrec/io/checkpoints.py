"""Feature-set registry + checkpoint / feature-matrix loaders.

This module owns:

- The named `FeatureSet` registry (v1, v4, …). Adding a new feature set is one
  entry here; downstream loaders auto-detect by `item_input_dim`.
- Loading a trained `TwoTowerModel` from disk in eval mode.
- Loading the on-disk feature matrices (item + bulk-user) that match a given
  checkpoint's input dims.
- A batched encoder used by FAISS index build, evaluation, and serving.

Everything here is **production-runtime** code — no eval-only sampling helpers
live here (see `eval_data.py` for those).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from mybookrec import DATA_DIR
from mybookrec.model.towers import TwoTowerModel


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
            bulk_user_npy: Filename under data/transformed/ for the full user
                matrix used during training.
            personal_user_npy: Filename for the single-row personal user vector.
            item_files: One or more filenames whose concatenation produces the
                item feature matrix.
        """
        self.name = name
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.bulk_user_npy = bulk_user_npy
        self.personal_user_npy = personal_user_npy
        self.item_files = item_files


# Paths are relative to `data/transformed/`. The two-segment leading dir encodes both the
# embedding-model run (`v1_minilm`, `v2_mxbai`, ...) and whether the artifact is
# model-independent (`shared/`) — see plans/book-recommender-mvp-plan.md for the full layout.
#
# Naming convention: <embedding_model>_<feature_variant>. `_basic` = no author features;
# unsuffixed = with author. Item dims for v2_mxbai assume embed_dim=512 (mxbai Matryoshka-
# truncated). If you change `embed_dim` in settings, regenerate the FeatureSet entries here.
FEATURE_SETS: tuple[FeatureSet, ...] = (
    FeatureSet(
        name="minilm_basic",
        user_input_dim=779,
        item_input_dim=395,
        bulk_user_npy="v1_minilm/train_user_features_basic.npy",
        personal_user_npy="v1_minilm/user_features_basic.npy",
        item_files=(
            "v1_minilm/book_embeddings.npy",
            "shared/genre_matrix.npy",
            "shared/num_pages_normalized.npy",
        ),
    ),
    FeatureSet(
        name="minilm_author",
        user_input_dim=1163,
        item_input_dim=779,
        bulk_user_npy="v1_minilm/train_user_features.npy",
        personal_user_npy="v1_minilm/user_features.npy",
        item_files=("v1_minilm/item_features.npy",),
    ),
    FeatureSet(
        name="mxbai_basic",
        # 512 (embed) + 10 (genre) + 1 (pages) = 523 item; +512 (dislike) = 1035 user.
        user_input_dim=1035,
        item_input_dim=523,
        bulk_user_npy="v2_mxbai/train_user_features_basic.npy",
        personal_user_npy="v2_mxbai/user_features_basic.npy",
        item_files=(
            "v2_mxbai/book_embeddings.npy",
            "shared/genre_matrix.npy",
            "shared/num_pages_normalized.npy",
        ),
    ),
    FeatureSet(
        name="mxbai_author",
        # 523 (basic item) + 512 (author emb) = 1035 item; 1035 (basic user) + 512 (author taste) = 1547 user.
        user_input_dim=1547,
        item_input_dim=1035,
        bulk_user_npy="v2_mxbai/train_user_features.npy",
        personal_user_npy="v2_mxbai/user_features.npy",
        item_files=("v2_mxbai/item_features.npy",),
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
        if all((transformed / path).exists() for path in fs.item_files):
            return fs
    return FEATURE_SETS[0]


def load_npy_to_device(path: Path, device: str) -> torch.Tensor:
    """Load a .npy file as a float32 torch tensor on the given device.

    Embeddings on disk may be stored as float16 to halve their footprint (this is the
    default for the Colab-produced MPNet/mxbai outputs since switching to Matryoshka +
    FP16 storage). We always upcast to float32 here because PyTorch math on MPS/CPU
    is consistently faster + more numerically stable in fp32, and the cast cost is
    negligible vs the size savings on disk + network.

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
