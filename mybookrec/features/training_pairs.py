"""PyTorch Dataset for two-tower training.

Emits (user_idx, positive_book_idx, negative_book_indices) per call. Negatives are sampled
fresh on every __getitem__ — across an epoch, the DataLoader iteration naturally produces
different negatives for the same positive, so the model can't memorize specific pairings.

Supports two negative-sampling distributions:
- "uniform": every book equally likely. Simplest, weakest baseline.
- "log_freq": p(book) ∝ log(interaction_count + 1). Popular books appear as negatives more
  often — this surfaces "hard negatives" (popular books that don't match user taste) so the
  model learns to discriminate by taste, not popularity.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from torch.utils.data import Dataset

from mybookrec import DATA_DIR


class TrainingPairsDataset(Dataset):
    """PyTorch Dataset emitting (user_idx, positive_book_idx, negative_book_indices) per call.

    Anchors are positive (rating ≥ 4) interactions from the requested data_split.
    Negatives are freshly sampled on every __getitem__ so the same positive sees
    different negatives across epochs without precomputing a static set.
    """

    def __init__(
        self,
        n_negatives: int = 4,
        data_split: str = "train",
        rng_seed: int | None = None,
        verbose: bool = True,
        negative_sampling: str = "uniform",
    ) -> None:
        """Load positives, exclusion sets, and (if log_freq) the sampling CDF.

        Args:
            n_negatives: Sampled negatives per anchor.
            data_split: One of "train", "validation", "test" — filters the source parquet.
            rng_seed: RNG seed (None for non-deterministic).
            verbose: Print progress lines during init (~10 s for the train split).
            negative_sampling: "uniform" or "log_freq" — see module docstring.
        """
        if negative_sampling not in ("uniform", "log_freq"):
            raise ValueError(f"negative_sampling must be 'uniform' or 'log_freq', got {negative_sampling!r}")

        def log(msg: str):
            if verbose:
                print(msg, flush=True)

        t0 = time.time()
        log(f"[TrainingPairsDataset/{data_split}] loading id mappings...")
        with open(DATA_DIR / "transformed" / "user_id_to_index.json") as f:
            user_id_to_index = json.load(f)
        with open(DATA_DIR / "transformed" / "book_id_to_index.json") as f:
            book_id_to_index = json.load(f)

        log(f"[TrainingPairsDataset/{data_split}] scanning interactions parquet... ({time.time() - t0:.1f}s)")
        interactions = (
            pl.scan_parquet(DATA_DIR / "transformed" / "training_interactions.parquet")
            .filter(pl.col("data_split") == data_split)
            .select("user_id", "book_id", "rating")
            .collect()
        )
        log(f"[TrainingPairsDataset/{data_split}]   loaded {len(interactions):,} rows ({time.time() - t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] mapping ids via polars join... ({time.time() - t0:.1f}s)")
        user_map = pl.DataFrame(
            {
                "user_id": list(user_id_to_index.keys()),
                "user_idx": list(user_id_to_index.values()),
            },
            schema={"user_id": pl.String, "user_idx": pl.Int64},
        )
        book_map = pl.DataFrame(
            {
                "book_id": list(book_id_to_index.keys()),
                "book_idx": list(book_id_to_index.values()),
            },
            schema={"book_id": pl.String, "book_idx": pl.Int64},
        )

        interactions = (
            interactions.with_columns(pl.col("book_id").cast(pl.String))
            .join(user_map, on="user_id", how="left")
            .join(book_map, on="book_id", how="left")
            .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
        )
        log(f"[TrainingPairsDataset/{data_split}]   {len(interactions):,} rows after mapping ({time.time() - t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] extracting positives... ({time.time() - t0:.1f}s)")
        positives = interactions.filter(pl.col("rating") >= 4)
        self.positive_users = positives["user_idx"].to_numpy()
        self.positive_books = positives["book_idx"].to_numpy()
        log(f"[TrainingPairsDataset/{data_split}]   {len(self.positive_users):,} positives ({time.time() - t0:.1f}s)")

        log(f"[TrainingPairsDataset/{data_split}] building per-user exclude sets... ({time.time() - t0:.1f}s)")
        exclude_groups = interactions.group_by("user_idx").agg(pl.col("book_idx").alias("rated_books"))
        self.exclude: dict[int, np.ndarray] = {
            row["user_idx"]: np.array(row["rated_books"], dtype=np.int64) for row in exclude_groups.to_dicts()
        }
        log(
            f"[TrainingPairsDataset/{data_split}]   {len(self.exclude):,} users in exclude "
            f"dict ({time.time() - t0:.1f}s)"
        )

        self.n_books = len(book_id_to_index)
        self.n_negatives = n_negatives
        self.rng = np.random.default_rng(rng_seed)
        self.negative_sampling = negative_sampling
        self.neg_sampling_cdf: np.ndarray | None = None

        if negative_sampling == "log_freq":
            log(f"[TrainingPairsDataset/{data_split}] computing log-frequency CDF... ({time.time() - t0:.1f}s)")
            # Count interactions per book across this split. Popular books get higher weight.
            book_counts = np.bincount(
                interactions["book_idx"].to_numpy().astype(np.int64),
                minlength=self.n_books,
            )
            # log(count + 1) gives positive weight to every book, even those with 0 interactions
            # (rare but possible if the book is in the catalog but only appears in val/test).
            weights = np.log1p(book_counts).astype(np.float64)
            probs = weights / weights.sum()
            # Inverse-CDF sampling: precompute the cumulative distribution once, then sample via
            # uniform-random + binary search. ~100x faster than torch.multinomial for many
            # small per-call draws (which is exactly the DataLoader hot path).
            self.neg_sampling_cdf = np.cumsum(probs)
            self.neg_sampling_cdf[-1] = 1.0  # pin to exactly 1 to absorb floating-point drift
            log(
                f"[TrainingPairsDataset/{data_split}]   probs: "
                f"min={probs.min():.2e} max={probs.max():.2e} "
                f"ratio_top_to_median={probs.max() / np.median(probs):.1f}x "
                f"({time.time() - t0:.1f}s)"
            )

        log(f"[TrainingPairsDataset/{data_split}] ready in {time.time() - t0:.1f}s")

    def __len__(self) -> int:
        """Return the number of positive anchors in this split."""
        return len(self.positive_users)

    def __getitem__(self, idx: int) -> tuple[int, int, np.ndarray]:
        """Return one (user_idx, positive_book_idx, n_negatives book indices) anchor.

        Args:
            idx: Anchor index in 0..len(self)-1.

        Returns:
            Triple of (user_idx, positive_book_idx, sampled_negative_book_indices).
            The negatives are freshly sampled per call.
        """
        user_idx = int(self.positive_users[idx])
        pos_book_idx = int(self.positive_books[idx])
        neg_book_indices = self._sample_negatives(user_idx)
        return user_idx, pos_book_idx, neg_book_indices

    def _draw_candidates(self, k: int) -> np.ndarray:
        if self.negative_sampling == "log_freq":
            # Inverse-CDF sampling: draw k uniforms in [0, 1), binary-search the CDF.
            r = self.rng.random(k)
            return np.minimum(np.searchsorted(self.neg_sampling_cdf, r), self.n_books - 1)
        return self.rng.integers(0, self.n_books, size=k)

    def _sample_negatives(self, user_idx: int) -> np.ndarray:
        excluded = self.exclude.get(user_idx)
        sampled = self._draw_candidates(self.n_negatives * 2)
        if excluded is not None and len(excluded) > 0:
            sampled = sampled[~np.isin(sampled, excluded)]
        while len(sampled) < self.n_negatives:
            extra = self._draw_candidates(self.n_negatives)
            if excluded is not None and len(excluded) > 0:
                extra = extra[~np.isin(extra, excluded)]
            sampled = np.concatenate([sampled, extra])
        return sampled[: self.n_negatives]
