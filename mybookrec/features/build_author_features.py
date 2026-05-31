"""Build author features for the with-author feature variant.

Pipeline:
1. Extract (book_id, primary_author_id) for the 1.78M catalog books.
2. Build author_id -> author_idx mapping.
3. Compute author embedding per author = mean of their books' description embeddings.
4. Build new (n_books, 779) item_features matrix: [book_emb | genre | pages | author_emb]
5. Build new (n_users, 1163) user_features matrix:
   [like_emb | dislike_emb | genre_dist | mean_pages | author_taste]
   where author_taste = rating-weighted mean of authors' embeddings for the user's 4+ star books.

Output files (under data/transformed/):
- shared/book_to_author_idx.npy           (n_books,) int array
- shared/author_id_to_index.json
- v1_minilm/author_embeddings.npy         (n_authors, 384)
- v1_minilm/item_features.npy             (n_books, 779)
- v1_minilm/train_user_features.npy       (n_users, 1163)
- v1_minilm/user_features.npy             (personal user, (1163,))
"""

import gzip
import json
import time

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from mybookrec import DATA_DIR
from mybookrec.settings import get_settings

t_total = time.time()
shared = DATA_DIR / "transformed" / "shared"
model_run_dir = DATA_DIR / "transformed" / get_settings().embed_model_run
model_run_dir.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """Print a message with the elapsed time since script start.

    Args:
        msg: The message to print.
    """
    print(f"[{time.time() - t_total:6.1f}s] {msg}", flush=True)


# Step 1: extract book_id -> primary_author_id.
log("Extracting primary author per book from raw json...")
with open(shared / "book_id_to_index.json") as f:
    book_id_to_index = json.load(f)

book_id_to_author: dict[str, str] = {}
with gzip.open(DATA_DIR / "raw" / "ucsd" / "goodreads_books.json.gz", "rt") as f:
    for line in f:
        b = json.loads(line)
        bid = b.get("book_id")
        if bid not in book_id_to_index:
            continue
        authors = b.get("authors") or []
        if not authors:
            continue
        # First author is the primary (role="" by Goodreads convention).
        book_id_to_author[bid] = authors[0]["author_id"]

log(f"Extracted authors for {len(book_id_to_author):,} of {len(book_id_to_index):,} books")

# Step 2: build author_id -> author_idx.
unique_authors = sorted(set(book_id_to_author.values()))
author_id_to_idx = {aid: i for i, aid in enumerate(unique_authors)}
n_authors = len(unique_authors)
log(f"n_authors = {n_authors:,}")

with open(shared / "author_id_to_index.json", "w") as f:
    json.dump(author_id_to_idx, f)

# Step 3: parallel array book_idx -> author_idx, -1 sentinel for books with no author.
n_books = len(book_id_to_index)
book_to_author_idx = np.full(n_books, -1, dtype=np.int64)
for bid, aid in book_id_to_author.items():
    book_to_author_idx[book_id_to_index[bid]] = author_id_to_idx[aid]

n_missing = int((book_to_author_idx == -1).sum())
log(f"Books with no author: {n_missing:,} of {n_books:,} ({n_missing / n_books:.1%})")
np.save(shared / "book_to_author_idx.npy", book_to_author_idx)

# Step 4: author embedding = mean of book embeddings written by this author.
# Sparse matmul: (n_authors x n_books) @ (n_books x 384) → sum, divide by per-author book count.
log("Loading book embeddings...")
book_embeddings = np.load(model_run_dir / "book_embeddings.npy").astype(np.float32)
embed_dim = book_embeddings.shape[1]
log(f"book_embeddings: {book_embeddings.shape} {book_embeddings.dtype}")

log("Computing author embeddings via sparse matmul...")
valid_books_mask = book_to_author_idx >= 0
valid_book_idxs = np.where(valid_books_mask)[0]
valid_author_idxs = book_to_author_idx[valid_books_mask]

author_book_matrix = csr_matrix(
    (np.ones(len(valid_book_idxs), dtype=np.float32), (valid_author_idxs, valid_book_idxs)),
    shape=(n_authors, n_books),
)

author_emb_sum = author_book_matrix @ book_embeddings
author_book_count = np.asarray(author_book_matrix.sum(axis=1)).flatten()
safe_count = np.where(author_book_count > 0, author_book_count, 1.0)
author_embeddings = (author_emb_sum / safe_count[:, None]).astype(np.float32)
log(f"author_embeddings: {author_embeddings.shape}")
log(
    f"  books-per-author distribution: mean={author_book_count.mean():.1f} "
    f"median={np.median(author_book_count):.0f} max={author_book_count.max():.0f}"
)
np.save(model_run_dir / "author_embeddings.npy", author_embeddings)

# Step 5: item_features_v4 = [book_emb | genre | pages | author_emb].
# Books with no author get a zero author embedding (no signal).
log("Building item_features (with author)...")
genre_matrix = np.load(shared / "genre_matrix.npy").astype(np.float32)
pages_vec = np.load(shared / "num_pages_normalized.npy").astype(np.float32)

book_author_emb = np.zeros((n_books, embed_dim), dtype=np.float32)
book_author_emb[valid_books_mask] = author_embeddings[book_to_author_idx[valid_books_mask]]

item_features = np.concatenate(
    [book_embeddings, genre_matrix, pages_vec.reshape(-1, 1), book_author_emb],
    axis=1,
).astype(np.float32)
log(f"item_features (with author): {item_features.shape}")
np.save(model_run_dir / "item_features.npy", item_features)

# Step 6: append author_taste to bulk user features.
# author_taste[u] = rating-weighted mean of book_author_emb over u's 4+ star books.
log("Loading training interactions for user feature rebuild...")
with open(shared / "user_id_to_index.json") as f:
    user_id_to_index = json.load(f)

user_map = pl.DataFrame(
    {"user_id": list(user_id_to_index.keys()), "user_idx": list(user_id_to_index.values())},
    schema={"user_id": pl.String, "user_idx": pl.Int64},
)
book_map = pl.DataFrame(
    {"book_id": list(book_id_to_index.keys()), "book_idx": list(book_id_to_index.values())},
    schema={"book_id": pl.String, "book_idx": pl.Int64},
)

interactions = (
    pl.scan_parquet(shared / "training_interactions.parquet")
    .filter(pl.col("data_split") == "train")
    .select("user_id", "book_id", "rating")
    .with_columns(pl.col("book_id").cast(pl.String))
    .join(user_map.lazy(), on="user_id", how="left")
    .join(book_map.lazy(), on="book_id", how="left")
    .filter(pl.col("user_idx").is_not_null() & pl.col("book_idx").is_not_null())
    .collect()
)
log(f"interactions: {len(interactions):,} rows")

user_idx = interactions["user_idx"].to_numpy()
book_idx = interactions["book_idx"].to_numpy()
rating = interactions["rating"].to_numpy().astype(np.float32)
liked_mask = rating >= 4

n_users = int(user_idx.max() + 1)

log("Computing per-user author taste via sparse matmul...")
W_like = csr_matrix(
    (rating[liked_mask], (user_idx[liked_mask], book_idx[liked_mask])),
    shape=(n_users, n_books),
)
weight_sum = np.asarray(W_like.sum(axis=1)).flatten()
safe_weight = np.where(weight_sum > 0, weight_sum, 1.0).astype(np.float32)
author_taste = ((W_like @ book_author_emb) / safe_weight[:, None]).astype(np.float32)
log(f"author_taste: {author_taste.shape}")

# Concatenate onto existing bulk user features. The compact mapping in user_id_to_index.json
# was built from these same interactions with user_idx in [0, n_old), so the index spaces
# line up and we can concat row-by-row.
log("Loading existing train_user_features and concatenating author_taste...")
old_train_user_features = np.load(model_run_dir / "train_user_features_basic.npy").astype(np.float32)
log(f"old_train_user_features: {old_train_user_features.shape}, recomputed author_taste: {author_taste.shape}")
assert author_taste.shape[0] == old_train_user_features.shape[0], (
    f"row mismatch: author_taste has {author_taste.shape[0]} users, old has {old_train_user_features.shape[0]}"
)

train_user_features = np.concatenate([old_train_user_features, author_taste], axis=1).astype(np.float32)
log(f"train_user_features (with author): {train_user_features.shape}")
np.save(model_run_dir / "train_user_features.npy", train_user_features)

# Step 7: personal user features v4 (same rating-weighted mean, single row).
log("Building personal user features (with author)...")
my_books = pl.read_csv(shared / "my_books.csv")
my_book_ids = my_books["book_id"].to_list()
my_ratings = my_books["my_rating"].to_numpy()
my_book_idxs = np.array([book_id_to_index.get(str(bid), -1) for bid in my_book_ids])
liked_personal = (my_ratings >= 4) & (my_book_idxs >= 0)

if liked_personal.sum() > 0:
    liked_idxs = my_book_idxs[liked_personal]
    liked_ratings_arr = my_ratings[liked_personal].astype(np.float32)
    personal_author_taste = (
        (book_author_emb[liked_idxs] * liked_ratings_arr[:, None]).sum(axis=0) / liked_ratings_arr.sum()
    ).astype(np.float32)
else:
    personal_author_taste = np.zeros(embed_dim, dtype=np.float32)

old_user_features = np.load(model_run_dir / "user_features_basic.npy").astype(np.float32)
user_features = np.concatenate([old_user_features, personal_author_taste], axis=0).astype(np.float32)
log(f"personal user_features (with author): {user_features.shape}")
np.save(model_run_dir / "user_features.npy", user_features)

log("DONE")
log(f"Total time: {time.time() - t_total:.0f}s")
