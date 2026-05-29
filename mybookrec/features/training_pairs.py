"""PyTorch Dataset for two-tower training.

Emits (user_idx, positive_book_idx, negative_book_indices) per call. Negatives are sampled
fresh on every __getitem__ — across an epoch, the DataLoader iteration naturally produces
different negatives for the same positive, so the model can't memorize specific pairings.

The Dataset only emits integer indices. Feature lookup against the precomputed
user_features / item_features matrices happens in the training loop.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from torch.utils.data import Dataset

from mybookrec import DATA_DIR


class TrainingPairsDataset(Dataset):
    def __init__(
        self,
        n_negatives: int = 4,
        data_split: str = "train",
        rng_seed: int | None = None,
        verbose: bool = True,
    ):
        # Progress prints keep VS Code / Colab websockets alive — init can take 1-3 min on Colab,
        # and silent processing causes the connection to time out.
        def log(msg: str):
            if verbose:
                print(msg, flush=True)

        t0 = time.time()
        log(f"[TrainingPairsDataset/{data_split}] loading id mappings...")
        with open(DATA_DIR / "transformed" / "user_id_to_index.json", "r") as f:
            user_id_to_index = json.load(f)
        with open(DATA_DIR / "transformed" / "book_id_to_index.json", "r") as f:
            book_id_to_index = json.load(f)

        log(f"[TrainingPairsDataset/{data_split}] scanning interactions parquet... ({time.time()-t0:.1f}s)")
        interactions = (
            pl.scan_parquet(DATA_DIR / "transformed" / "training_interactions.parquet")
            .filter(pl.col("data_split") == data_split)
            .select("user_id", "book_id", "rating")
            .collect()
        )
        log(f"[TrainingPairsDataset/{data_split}]   loaded {len(interactions):,} rows ({time.time()-t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] mapping ids via polars join... ({time.time()-t0:.1f}s)")
        # Using native polars joins is ~10x faster than per-row map_elements lambdas:
        # vectorized hash join in C vs a Python dict lookup per row over ~70M rows.
        user_map = pl.DataFrame({
            "user_id": list(user_id_to_index.keys()),
            "user_idx": list(user_id_to_index.values()),
        }, schema={"user_id": pl.String, "user_idx": pl.Int64})
        book_map = pl.DataFrame({
            "book_id": list(book_id_to_index.keys()),
            "book_idx": list(book_id_to_index.values()),
        }, schema={"book_id": pl.String, "book_idx": pl.Int64})

        interactions = (
            interactions
            .with_columns(pl.col("book_id").cast(pl.String))
            .join(user_map, on="user_id", how="left")
            .join(book_map, on="book_id", how="left")
            .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
        )
        log(f"[TrainingPairsDataset/{data_split}]   {len(interactions):,} rows after mapping ({time.time()-t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] extracting positives... ({time.time()-t0:.1f}s)")
        positives = interactions.filter(pl.col("rating") >= 4)
        self.positive_users = positives["user_idx"].to_numpy()
        self.positive_books = positives["book_idx"].to_numpy()
        log(f"[TrainingPairsDataset/{data_split}]   {len(self.positive_users):,} positives ({time.time()-t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] building per-user exclude sets... ({time.time()-t0:.1f}s)")
        exclude_groups = (
            interactions.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
        )
        self.exclude: dict[int, np.ndarray] = {
            row["user_idx"]: np.array(row["rated_books"], dtype=np.int64)
            for row in exclude_groups.to_dicts()
        }
        log(f"[TrainingPairsDataset/{data_split}]   {len(self.exclude):,} users in exclude dict ({time.time()-t0:.1f}s)")

        self.n_books = len(book_id_to_index)
        self.n_negatives = n_negatives
        self.rng = np.random.default_rng(rng_seed)
        log(f"[TrainingPairsDataset/{data_split}] ready in {time.time()-t0:.1f}s")

    def __len__(self) -> int:
        return len(self.positive_users)

    def __getitem__(self, idx: int):
        user_idx = int(self.positive_users[idx])
        pos_book_idx = int(self.positive_books[idx])
        neg_book_indices = self._sample_negatives(user_idx)
        return user_idx, pos_book_idx, neg_book_indices

    def _sample_negatives(self, user_idx: int) -> np.ndarray:
        excluded = self.exclude.get(user_idx)
        sampled = self.rng.integers(0, self.n_books, size=self.n_negatives * 2)
        if excluded is not None and len(excluded) > 0:
            sampled = sampled[~np.isin(sampled, excluded)]
        while len(sampled) < self.n_negatives:
            extra = self.rng.integers(0, self.n_books, size=self.n_negatives)
            if excluded is not None and len(excluded) > 0:
                extra = extra[~np.isin(extra, excluded)]
            sampled = np.concatenate([sampled, extra])
        return sampled[:self.n_negatives]
