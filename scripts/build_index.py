"""Build a FAISS index of item embeddings from a trained checkpoint.

Usage:
    .venv/bin/python scripts/build_index.py checkpoints/two_tower_v4bce_best.pt
    .venv/bin/python scripts/build_index.py <ckpt> --output checkpoints/book_index_v4.faiss
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# macOS libomp conflict between FAISS and PyTorch — documented workaround.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from mybookrec import DATA_DIR
from mybookrec.index import build_index, encode_all_items
from mybookrec.io import load_checkpoint, load_item_features


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace with `checkpoint` and `output` attributes.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .faiss path. Defaults to checkpoints/<checkpoint_stem>_index.faiss.",
    )
    return p.parse_args()


def main() -> None:
    """Build and serialize the FAISS item index for the supplied checkpoint."""
    args = parse_args()

    model, config, _ = load_checkpoint(args.checkpoint)
    print(
        f"Checkpoint: {args.checkpoint}  item_input_dim={config['item_input_dim']}  "
        f"embedding_dim={config['hidden_dims'][-1]}"
    )

    device = next(model.parameters()).device.type
    item_features = load_item_features(config, device)
    print(f"Item features: {tuple(item_features.shape)}")

    t0 = time.time()
    item_embeddings = encode_all_items(model, item_features)
    print(f"Encoded {item_embeddings.shape[0]:,} items to {item_embeddings.shape[1]}-dim in {time.time() - t0:.1f}s")

    output = args.output or (DATA_DIR.parent / "checkpoints" / f"{args.checkpoint.stem}_index.faiss")
    t0 = time.time()
    build_index(item_embeddings, save_path=output)
    print(f"Index saved to {output}  ({output.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
