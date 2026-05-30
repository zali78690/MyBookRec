"""Build author features for v4.

Pipeline:
1. Extract (book_id, primary_author_id) for the 1.78M catalog books.
2. Build author_id -> author_idx mapping.
3. Compute author embedding per author = mean of their books' description embeddings.
4. Build new (n_books, 779) item_features matrix: [book_emb | genre | pages | author_emb]
5. Build new (n_users, 1163) user_features matrix:
   [like_emb | dislike_emb | genre_dist | mean_pages | author_taste]
   where author_taste = rating-weighted mean of authors' embeddings for the user's 4+ star books.

Output files:
- data/transformed/book_to_author_idx.npy   (n_books,) int array
- data/transformed/author_embeddings.npy    (n_authors, 384)
- data/transformed/item_features_v4.npy     (n_books, 779)
- data/transformed/train_user_features_v4.npy (n_users, 1163)
- data/transformed/user_features_v4.npy     (personal user, (1163,))
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from mybookrec import DATA_DIR

t_total = time.time()
transformed = DATA_DIR / "transformed"


def log(msg):
    print(f"[{time.time()-t_total:6.1f}s] {msg}", flush=True)


# Step 1: extract book_id -> primary_author_id
log("Extracting primary author per book from raw json...")
with open(transformed / "book_id_to_index.json") as f:
    book_id_to_index = json.load(f)

book_id_to_author: dict[str, str] = {}
with gzip.open(DATA_DIR / "raw" / "goodreads_books.json.gz", "rt") as f:
    for line in f:
        b = json.loads(line)
        bid = b.get("book_id")
        if bid not in book_id_to_index:
            continue
        authors = b.get("authors") or []
        if not authors:
            continue
        # Take the first author as primary (role="" is the main author)
        book_id_to_author[bid] = authors[0]["author_id"]

log(f"Extracted authors for {len(book_id_to_author):,} of {len(book_id_to_index):,} books")

# Step 2: build author_id -> author_idx
unique_authors = sorted(set(book_id_to_author.values()))
author_id_to_idx = {aid: i for i, aid in enumerate(unique_authors)}
n_authors = len(unique_authors)
log(f"n_authors = {n_authors:,}")

# Save author mapping
with open(transformed / "author_id_to_index.json", "w") as f:
    json.dump(author_id_to_idx, f)

# Step 3: book_idx -> author_idx (parallel arrays, indexed by book_idx)
n_books = len(book_id_to_index)
book_to_author_idx = np.full(n_books, -1, dtype=np.int64)
for bid, aid in book_id_to_author.items():
    b_idx = book_id_to_index[bid]
    book_to_author_idx[b_idx] = author_id_to_idx[aid]

n_missing = int((book_to_author_idx == -1).sum())
log(f"Books with no author: {n_missing:,} of {n_books:,} ({n_missing/n_books:.1%})")
np.save(transformed / "book_to_author_idx.npy", book_to_author_idx)

# Step 4: compute author embedding = mean of book embeddings written by this author
log("Loading book embeddings...")
book_embeddings = np.load(transformed / "book_embeddings.npy").astype(np.float32)
embed_dim = book_embeddings.shape[1]
log(f"book_embeddings: {book_embeddings.shape} {book_embeddings.dtype}")

# For each author, sum the embeddings of their books. Use a sparse matrix for speed.
log("Computing author embeddings via sparse matmul...")
valid_books_mask = book_to_author_idx >= 0
valid_book_idxs = np.where(valid_books_mask)[0]
valid_author_idxs = book_to_author_idx[valid_books_mask]

# (n_authors, n_books) sparse matrix where M[a, b] = 1 if book b has author a
author_book_matrix = csr_matrix(
    (np.ones(len(valid_book_idxs), dtype=np.float32),
     (valid_author_idxs, valid_book_idxs)),
    shape=(n_authors, n_books),
)

# author_emb_sum (n_authors, 384), author_book_count (n_authors,)
author_emb_sum = author_book_matrix @ book_embeddings  # (n_authors, 384)
author_book_count = np.asarray(author_book_matrix.sum(axis=1)).flatten()
safe_count = np.where(author_book_count > 0, author_book_count, 1.0)
author_embeddings = (author_emb_sum / safe_count[:, None]).astype(np.float32)
log(f"author_embeddings: {author_embeddings.shape}")
log(f"  books-per-author distribution: mean={author_book_count.mean():.1f} median={np.median(author_book_count):.0f} max={author_book_count.max():.0f}")
np.save(transformed / "author_embeddings.npy", author_embeddings)

# Step 5: build item_features_v4 = [book_emb | genre | pages | author_emb]
log("Building item_features_v4...")
genre_matrix = np.load(transformed / "genre_matrix.npy").astype(np.float32)
pages_vec = np.load(transformed / "num_pages_normalized.npy").astype(np.float32)

# For books with no author (book_to_author_idx == -1), use zero author embedding (no signal)
book_author_emb = np.zeros((n_books, embed_dim), dtype=np.float32)
book_author_emb[valid_books_mask] = author_embeddings[book_to_author_idx[valid_books_mask]]

item_features_v4 = np.concatenate(
    [book_embeddings, genre_matrix, pages_vec.reshape(-1, 1), book_author_emb],
    axis=1,
).astype(np.float32)
log(f"item_features_v4: {item_features_v4.shape}")
np.save(transformed / "item_features_v4.npy", item_features_v4)

# Step 6: rebuild bulk user features with author_taste appended
# author_taste[u] = rating-weighted mean of book_author_emb for u's 4+ star books
log("Loading training interactions for user feature rebuild...")
with open(transformed / "user_id_to_index.json") as f:
    user_id_to_index = json.load(f)

user_map = pl.DataFrame({"user_id": list(user_id_to_index.keys()), "user_idx": list(user_id_to_index.values())}, schema={"user_id": pl.String, "user_idx": pl.Int64})
book_map = pl.DataFrame({"book_id": list(book_id_to_index.keys()), "book_idx": list(book_id_to_index.values())}, schema={"book_id": pl.String, "book_idx": pl.Int64})

interactions = (
    pl.scan_parquet(transformed / "training_interactions.parquet")
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

# Weighted matmul: author_taste[u] = sum(rating * book_author_emb[book]) / sum(rating)
# over books where u rated 4+ stars.
log("Computing per-user author taste via sparse matmul...")
W_like = csr_matrix(
    (rating[liked_mask],
     (user_idx[liked_mask], book_idx[liked_mask])),
    shape=(n_users, n_books),
)
weight_sum = np.asarray(W_like.sum(axis=1)).flatten()
safe_weight = np.where(weight_sum > 0, weight_sum, 1.0).astype(np.float32)
author_taste = ((W_like @ book_author_emb) / safe_weight[:, None]).astype(np.float32)
log(f"author_taste: {author_taste.shape}")

# Concatenate onto the existing train_user_features
log("Loading existing train_user_features and concatenating author_taste...")
old_train_user_features = np.load(transformed / "train_user_features.npy").astype(np.float32)
# Note: old_train_user_features may have a different n_users than the recomputed one (filtered).
# Align: old features only include users who passed the "has liked books" filter. Use the saved
# user_id_to_index.json which IS the compact mapping for that filtered set.
log(f"old_train_user_features: {old_train_user_features.shape}, recomputed author_taste: {author_taste.shape}")
assert author_taste.shape[0] >= old_train_user_features.shape[0], "user count mismatch — author_taste should cover all users"
# Take the rows corresponding to the compact mapping. The compact mapping was built from
# the same interactions, so user_idx values are valid into author_taste.
n_old = old_train_user_features.shape[0]
# Build inverse: compact_user_idx -> original_user_idx from user_id_to_index.json
compact_user_idxs = np.array(sorted(user_id_to_index.values()), dtype=np.int64)
# Verify: compact_user_idxs should equal arange(n_old) since values were assigned 0..n_old-1
assert (compact_user_idxs == np.arange(n_old)).all(), "compact mapping should be 0..n_old-1"

# Slice author_taste to match the compact user set
# But author_taste was indexed by the FULL n_users from train interactions, which equals
# the original (pre-compact) user_idx space. Since the compact mapping IS the original
# (we never reindexed when filtering to has-liked-books), the slicing is trivial.
# To verify: check shape
if author_taste.shape[0] != n_old:
    log(f"WARNING: shape mismatch — author_taste has {author_taste.shape[0]} users, old has {n_old}.")
    log("This means the original v3b user feature build dropped users between bulk-compute and save.")
    log("Slicing author_taste to first n_old rows under the assumption indices are preserved.")
    author_taste = author_taste[:n_old]

train_user_features_v4 = np.concatenate([old_train_user_features, author_taste], axis=1).astype(np.float32)
log(f"train_user_features_v4: {train_user_features_v4.shape}")
np.save(transformed / "train_user_features_v4.npy", train_user_features_v4)

# Step 7: build personal user features v4
log("Building personal user features v4...")
my_books = pl.read_csv(transformed / "my_books.csv")
# Map book_id -> book_idx
my_book_ids = my_books["book_id"].to_list()
my_ratings = my_books["my_rating"].to_numpy()
my_book_idxs = np.array([book_id_to_index.get(str(bid), -1) for bid in my_book_ids])
valid = my_book_idxs >= 0
liked_personal = (my_ratings >= 4) & valid

if liked_personal.sum() > 0:
    liked_idxs = my_book_idxs[liked_personal]
    liked_ratings_arr = my_ratings[liked_personal].astype(np.float32)
    personal_author_taste = (book_author_emb[liked_idxs] * liked_ratings_arr[:, None]).sum(axis=0) / liked_ratings_arr.sum()
    personal_author_taste = personal_author_taste.astype(np.float32)
else:
    personal_author_taste = np.zeros(embed_dim, dtype=np.float32)

old_user_features = np.load(transformed / "user_features.npy").astype(np.float32)
user_features_v4 = np.concatenate([old_user_features, personal_author_taste], axis=0).astype(np.float32)
log(f"personal user_features_v4: {user_features_v4.shape}")
np.save(transformed / "user_features_v4.npy", user_features_v4)

log("DONE")
log(f"Total time: {time.time()-t_total:.0f}s")
