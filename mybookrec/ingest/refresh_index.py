"""Incrementally append new gold items into a live FAISS index.

Reads gold/item_features.npy + gold/book_ids.json (produced by `to_gold.run`), encodes them
through the model's ItemTower (so embeddings match the existing index space), and appends
to the FAISS index in place. Also extends the on-disk book_id_to_index mapping so the new
items are addressable at serving time.

Safe to run multiple times: items whose book_id is already in the mapping are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# macOS libomp conflict between FAISS and PyTorch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from mybookrec.index.faiss_index import encode_all_items, load_index
from mybookrec.io.artifacts import TransformedArtifacts
from mybookrec.io.checkpoints import load_checkpoint
from mybookrec.settings import get_settings


def filter_new_items(
    gold_ids: list[str],
    gold_features: np.ndarray,
    existing_mapping: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    """Drop gold items already present in the existing mapping.

    Args:
        gold_ids: book_ids in gold (length n_gold).
        gold_features: (n_gold, item_feature_dim) feature matrix.
        existing_mapping: Existing book_id → index dict.

    Returns:
        Tuple of (new_ids, new_features) filtered to unseen books only.
    """
    keep_rows = [i for i, book_id in enumerate(gold_ids) if book_id not in existing_mapping]
    new_ids = [gold_ids[i] for i in keep_rows]
    new_features = gold_features[keep_rows] if keep_rows else np.zeros((0, gold_features.shape[1]), dtype=np.float32)
    return new_ids, new_features


def run(
    index_path: Path,
    checkpoint_path: Path | None = None,
    artifacts: TransformedArtifacts | None = None,
) -> int:
    """Append all unseen gold items to the FAISS index and extend the id mapping.

    Args:
        index_path: Path to the FAISS index (will be overwritten with the appended version).
        checkpoint_path: Model checkpoint. Defaults to settings.resolved_serve_model_path().
        artifacts: TransformedArtifacts source for the book_id mapping (loaded + saved here).
            Defaults to one bound at settings.transformed_dir.

    Returns:
        Number of items appended.

    Raises:
        FileNotFoundError: If gold artifacts are missing.
        ValueError: If gold features have a dim other than the checkpoint's item_input_dim.
    """
    settings = get_settings()
    if checkpoint_path is None:
        checkpoint_path = settings.resolved_serve_model_path()
    if artifacts is None:
        artifacts = TransformedArtifacts(settings.transformed_dir)

    gold_features_path = settings.gold_dir / "item_features.npy"
    gold_ids_path = settings.gold_dir / "book_ids.json"
    if not gold_features_path.exists() or not gold_ids_path.exists():
        raise FileNotFoundError(f"Gold artifacts missing — run `ingest.cli gold` first ({settings.gold_dir})")

    gold_features = np.load(gold_features_path).astype(np.float32)
    with open(gold_ids_path) as f:
        gold_ids: list[str] = json.load(f)

    mapping = dict(artifacts.book_id_to_index)
    new_ids, new_features = filter_new_items(gold_ids, gold_features, mapping)
    if not new_ids:
        return 0

    model, config, _ = load_checkpoint(checkpoint_path)
    if new_features.shape[1] != config["item_input_dim"]:
        raise ValueError(
            f"Gold feature dim {new_features.shape[1]} != checkpoint item_input_dim {config['item_input_dim']}"
        )

    device = next(model.parameters()).device.type
    feature_tensor = torch.from_numpy(new_features).to(device).float()
    new_embeddings = encode_all_items(model, feature_tensor)

    index = load_index(index_path)
    if index.d != new_embeddings.shape[1]:
        raise ValueError(f"Index dim {index.d} != embedding dim {new_embeddings.shape[1]}")
    index.add(new_embeddings)

    import faiss

    faiss.write_index(index, str(index_path))

    next_idx = max(mapping.values()) + 1 if mapping else 0
    for book_id in new_ids:
        mapping[book_id] = next_idx
        next_idx += 1
    artifacts.save_book_id_to_index(mapping)

    return len(new_ids)
