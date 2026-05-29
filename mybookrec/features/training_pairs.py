"""PyTorch Dataset for two-tower training.

Emits (user_idx, positive_book_idx, negative_book_indices) per call. Negatives are sampled
fresh on every __getitem__ — across an epoch, the DataLoader iteration naturally produces
different negatives for the same positive, so the model can't memorize specific pairings.

The Dataset only emits integer indices. Feature lookup against the precomputed
user_features / item_features matrices happens in the training loop.
"""
from __future__ import annotations

import json

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
    ):
        with open(DATA_DIR / "transformed" / "user_id_to_index.json", "r") as f:
            user_id_to_index = json.load(f)
        with open(DATA_DIR / "transformed" / "book_id_to_index.json", "r") as f:
            book_id_to_index = json.load(f)

        interactions = (
            pl.scan_parquet(DATA_DIR / "transformed" / "books_with_interactions.parquet")
            .filter(pl.col("data_split") == data_split)
            .select("user_id", "book_id", "rating")
            .collect()
        )

        interactions = (
            interactions
            .with_columns(pl.col("book_id").cast(pl.String))
            .with_columns(
                pl.col("user_id").map_elements(
                    lambda u: user_id_to_index.get(u, -1), return_dtype=pl.Int64
                ).alias("user_idx"),
                pl.col("book_id").map_elements(
                    lambda b: book_id_to_index.get(b, -1), return_dtype=pl.Int64
                ).alias("book_idx"),
            )
            .filter((pl.col("user_idx") >= 0) & (pl.col("book_idx") >= 0))
        )

        positives = interactions.filter(pl.col("rating") >= 4)
        self.positive_users = positives["user_idx"].to_numpy()
        self.positive_books = positives["book_idx"].to_numpy()

        exclude_groups = (
            interactions.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
        )
        self.exclude: dict[int, np.ndarray] = {
            row["user_idx"]: np.array(row["rated_books"], dtype=np.int64)
            for row in exclude_groups.to_dicts()
        }

        self.n_books = len(book_id_to_index)
        self.n_negatives = n_negatives
        self.rng = np.random.default_rng(rng_seed)

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
