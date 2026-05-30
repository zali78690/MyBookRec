"""Build a FAISS index of all item embeddings from a trained checkpoint.

Loads the model, runs every book through the ItemTower, and saves the index for inference.

Usage:
    uv run python scripts/build_index.py checkpoints/two_tower_v4bce_best.pt
    uv run python scripts/build_index.py <path> --output checkpoints/book_index_v4.faiss
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# macOS libomp conflict between FAISS and PyTorch — documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from mybookrec import DATA_DIR
from mybookrec.index import build_index, encode_all_items
from mybookrec.model.towers import TwoTowerModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="Output .faiss path. Defaults to checkpoints/<checkpoint_stem>_index.faiss.")
    return p.parse_args()


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_item_features(config: dict, device: str) -> torch.Tensor:
    t = DATA_DIR / "transformed"
    if config["item_input_dim"] == 779:
        return torch.from_numpy(np.load(t / "item_features_v4.npy")).to(device).float()
    emb = torch.from_numpy(np.load(t / "book_embeddings.npy")).to(device).float()
    genre = torch.from_numpy(np.load(t / "genre_matrix.npy")).to(device).float()
    pages = torch.from_numpy(np.load(t / "num_pages_normalized.npy")).to(device).float().reshape(-1, 1)
    return torch.cat([emb, genre, pages], dim=1).contiguous()


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"  feature_set={config.get('feature_set', '?')}  "
          f"item_input_dim={config['item_input_dim']}  embedding_dim={config['hidden_dims'][-1]}")

    model = TwoTowerModel(
        user_input_dim=config["user_input_dim"],
        item_input_dim=config["item_input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    item_features = load_item_features(config, device)
    print(f"Item features: {tuple(item_features.shape)}")

    t0 = time.time()
    item_embs = encode_all_items(model, item_features)
    print(f"Encoded {item_embs.shape[0]:,} items to {item_embs.shape[1]}-dim in {time.time() - t0:.1f}s")

    output = args.output or (DATA_DIR.parent / "checkpoints" / f"{args.checkpoint.stem}_index.faiss")
    t0 = time.time()
    build_index(item_embs, save_path=output)
    print(f"Index saved to {output}  ({output.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
