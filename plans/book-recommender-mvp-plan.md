# MyBookRec — Implementation Plan and Current Status

## Problem statement

Goodreads recommendations are popularity-driven. As an active rater/shelver I want a system that learns *my* taste — what genres, lengths, prose styles I gravitate toward — and surfaces books I'd actually enjoy, not just what's trending.

Goodreads uses pure collaborative filtering ("users who rated the same books as you also liked these"). It can't recommend a new niche book that no similar user has rated yet.

This system is **content + behavior hybrid**:
- It understands what a book *is about* (description embeddings, genre, author).
- It learns *your* taste profile from your rating behavior.
- It can recommend books with zero ratings from similar users, as long as the content matches your learned taste.

## Solution

A two-tower neural network trained on the UCSD Goodreads dataset. The system encodes users and books into a shared embedding space; at inference, the user embedding is dot-producted against all book embeddings to retrieve top-K recommendations.

MVP scope: profile-based recommendations for a single user (me).

## Current status

| Component | State |
|---|---|
| Data pipeline | ✅ Done |
| EDA | ✅ Done |
| Item features (v1: 395-dim, v4: 779-dim with authors) | ✅ Done |
| User features (personal + bulk train, v1 779-dim, v4 1163-dim) | ✅ Done |
| Training pairs Dataset | ✅ Done |
| TwoTowerModel (`mybookrec/model/towers.py`) | ✅ Done |
| Training loop | ✅ Done (Mac MPS, ~14 batches/sec) |
| Eval metrics (HR@K, NDCG@K) | ✅ Done |
| InfoNCE loss (`mybookrec/model/loss.py`) | ⚠️ Built, didn't converge in budget |
| Cross-encoder reranker (`mybookrec/model/cross_encoder.py`) | ⚠️ Built, undertrained |
| FAISS index for inference | ⏳ Not yet built |
| Recommendation pipeline (`recommend.py`) | ⏳ Not yet built (manual via notebooks) |
| Personal vibe check notebook | ✅ Done |
| HF Hub artifact upload | ⏳ Not yet done |

### Current best model

**`data/checkpoints/two_tower_mac.pt` (v1):** BCE + uniform negative sampling + dropout=0.1. Trained ~56,000 batches on Apple Silicon MPS.

**Eval (5,000 test pairs, leave-one-out, rng=0)**:
- HR@10 = **0.0100** (1,667× random baseline)
- NDCG@10 = **0.0043**
- NDCG given hit = 0.43 (rank ~3-4 on average when hit)

## Architecture

### Two-tower retrieval

```
User features (779 or 1163-dim)        Item features (395 or 779-dim)
        ↓                                       ↓
    UserTower MLP                          ItemTower MLP
   779/1163 → 512 → 256 → 128            395/779 → 512 → 256 → 128
        ↓                                       ↓
   L2-normalize                          L2-normalize
        ↓                                       ↓
   user_emb (128)                         item_emb (128)
                    ↘                  ↙
                   dot product × τ
                          ↓
                   logit → sigmoid → BCE
```

- Dropout=0.1 between layers, ReLU activation.
- Learnable temperature `τ` (init=10) scales L2-normed dot product into a useful logit range.
- BCE loss with 4:1 negative-to-positive sampling.
- Adam optimizer, LR=1e-3.

### Items features

| Channel | Dim | Source |
|---|---|---|
| Description embedding | 384 | `ibm-granite/granite-embedding-30m-english` on `"{title}. {description}"` |
| Genre vector | 10 | L2-normalized count over fixed vocab (10 categories from `book_genres_initial.json`) |
| Normalized pages | 1 | Min-max with 1st/99th percentile clipping, median-imputed for missing |
| Author embedding (v4) | 384 | Mean of all books' description embeddings by that author |

v1 total: **395-dim**. v4 with author features: **779-dim**.

### User features

Built from train-split interactions only (no leakage from val/test).

| Channel | Dim | Source |
|---|---|---|
| Like embedding | 384 | Rating-weighted mean of liked books' (≥4 star) description embeddings |
| Dislike embedding | 384 | Simple mean of disliked books' (1-2 star) description embeddings (zero if none) |
| Genre distribution | 10 | L2-normalized sum of liked books' genre vectors |
| Mean pages | 1 | Average of liked books' normalized pages |
| Author taste (v4) | 384 | Rating-weighted mean of liked authors' embeddings |

v1 total: **779-dim**. v4 with author features: **1163-dim**.

## Data pipeline

1. **Raw data** — UCSD Goodreads JSON dumps (gitignored, ~30 GB across all files).
2. **Filter** — language ∈ `{eng, en-US, en-GB, ""}`, ratings_count ≥ 5 (books), explicit ratings only (drop rating=0), users with ≥10 ratings.
3. **Train labels** — rating ≥ 4 → positive, rating ∈ {1, 2} → negative, rating = 3 → excluded.
4. **Temporal per-user split** — 70% train / 10% val / 20% test by `date_added`.

**Slim parquet trick**: the full `books_with_interactions.parquet` is 19 GB (duplicates book metadata per interaction). For training we use a slimmed version `training_interactions.parquet` with only `(user_id, book_id, rating, data_split)` → ~530 MB, zstd-compressed. Built by [build_training_interactions.ipynb](../mybookrec/data_load/build_training_interactions.ipynb).

## Training pipeline

Implemented in [mybookrec/model/train.ipynb](../mybookrec/model/train.ipynb).

- Loads precomputed feature matrices to GPU (CUDA, MPS, or CPU).
- Iterates `TrainingPairsDataset` via DataLoader.
- Per batch: encode user (B, H), encode positive + 4 negatives (B, 5, H), dot-product, BCE.
- Checkpoints every 5K batches, early stopping with patience=5 on val loss.
- ~14 batches/sec on Apple Silicon MPS.

**Negative sampling** (in `TrainingPairsDataset`):
- `"uniform"` (default): random over the catalog with rejection of user-rated books.
- `"log_freq"`: probability ∝ log(book_count + 1). Implemented via inverse-CDF + `np.searchsorted` for speed (~100× faster than `torch.multinomial`).

## Eval pipeline

Implemented in [mybookrec/eval/](../mybookrec/eval/).

- **Metrics** (`metrics.py`): batched `hit_rate_at_k`, `ndcg_at_k`. Unit-tested.
- **Full eval** (`run_eval.ipynb`): samples 5,000 test (user, positive_book) pairs with fixed seed, scores all 1.78M books per user, masks train-rated books, computes HR@10 + NDCG@10.
- **Vibe check** (`vibe_check.ipynb`): generates top-20 recommendations for *me* against any trained checkpoint. Shows titles, average ratings, page counts, similarity scores.

## v3 overhaul attempts and findings

Detailed in [v3-overhaul-results.md](v3-overhaul-results.md). Summary:

| Experiment | What changed vs v1 | HR@10 | Verdict |
|---|---|---|---|
| v2 | log_freq sampling, dropout=0.3, weight_decay | 0.0066 | over-regularized |
| v3 (InfoNCE + in-batch + collision mask) | BCE → InfoNCE | 0.0015 → 0 | learnable τ diverged; didn't converge in budget even with fixed τ |
| v4-BCE (author features + BCE) | +author dim | 0.0062 | undertrained (5k vs v1's 56k batches; killed early due to system load) |
| v6 pipeline (v4-BCE + cross-encoder rerank) | retrieve top-100, rerank | 0.0054 | both stages undertrained; reranking at noise floor |

**Key bugs caught and fixed during overhaul** (preserved in `loss.py`):
1. In-batch positive leakage — when two users in a batch share the same positive book, the in-batch trick treats one as the other's negative. Fix: mask off-diagonal entries where `pos_idx[i] == pos_idx[j]`.
2. NaN gradients from `-inf × temperature` in autograd. Fix: mask after scaling by temperature, not before.

**Lesson**: training budget compounded — v1 had 11× more batches than the new architectures. None of the new architectures got enough training to validate their lift.

## Remaining MVP work

In rough priority order:

1. **Re-train v4-BCE for the full ~60-minute MPS budget** that v1 got. Most likely path to beat HR@10 = 0.0100.
2. **FAISS index** — load trained ItemTower, encode all 1.78M books, build `IndexFlatIP`, serialize. Today eval uses raw `torch.topk` over the full catalog (sub-second), but a saved FAISS index makes inference reproducible and portable.
3. **Recommendation CLI** (`scripts/recommend.py`) — wraps the vibe-check notebook as a reusable command-line tool. Args: checkpoint path, user CSV, top-K.
4. **Post-ranking filters** — `min_avg_rating`, `ebook_only` toggles applied to FAISS top-(K*3) before final top-K. Already-read books always excluded.
5. **HF Hub upload** — model checkpoint, item feature matrix, book_id mapping, genre vocab, author embeddings.
6. **Cloud training notebook** (`notebooks/cloud_train.ipynb`) — Colab T4 setup that runs `embed.ipynb` and the training script. Useful when Mac MPS runs become impractical.

## Deferred improvements (post-MVP)

Concrete next steps for each are in [v3-overhaul-deferred-items.md](v3-overhaul-deferred-items.md).

| Improvement | Expected lift | Effort | Notes |
|---|---|---|---|
| Better embeddings (MPNet 768 / BGE 1024-dim) | 1.2-1.3× | 4 GPU-hours | Single biggest content-side lever; benefits v1 immediately, no retraining of features |
| Review text features | 1.2-1.5× | 30 GPU-hours + data download | UCSD reviews ~50 GB uncompressed |
| BISAC / hybrid taxonomy | 1.2-1.5× | 4 hours data work | Replace 10-genre vocab with shelf-name embedding mapping to ~50 BISAC categories |
| SASRec (transformer over user sequence) | 1.5-2× | 1 week | Major architectural rewrite; captures temporal taste shifts |
| Hard negative mining | 1.5-2× | 1 hour | Code already in `/tmp/train_mac_v5.py`; only useful with a converged InfoNCE base |
| Temporal decay in user features | 1.2× | 30 min | Requires propagating `date_added` into slim parquet |

Also deferred:
- Dislike genre vector in user tower
- Inverse-rating weighting for dislike embedding
- Book-to-book similarity (trivial once item tower is trained)
- MLflow experiment tracking
- FAISS `IndexIVFFlat` (approximate search — unnecessary at 1.78M scale)
- Multi-user serving / web interface
- Pydantic `BaseSettings` + YAML config (currently hyperparams live in training scripts)

## Project structure (actual, as of v3 overhaul)

```
mybookrec/
├── __init__.py              # ROOT_DIR, DATA_DIR (env-var overrideable via MYBOOKREC_DATA_DIR)
├── data_load/
│   ├── transform_and_save.ipynb       # raw json → cleaned parquets + temporal split
│   └── build_training_interactions.ipynb  # slim parquet for fast training
├── features/
│   ├── generate_vocab.ipynb           # genre_vocab.json
│   ├── genre_vocab.json               # fixed 10-genre vocab
│   ├── build_item_feature.ipynb       # 395-dim item matrix
│   ├── build_user_feature.ipynb       # personal user vector
│   ├── build_train_user_features.ipynb # bulk user matrix for training
│   ├── build_training_pairs.ipynb     # Dataset walkthrough
│   └── training_pairs.py              # TrainingPairsDataset (importable)
├── model/
│   ├── towers.py                      # ItemTower, UserTower, TwoTowerModel
│   ├── loss.py                        # info_nce_in_batch (with collision mask)
│   ├── cross_encoder.py               # CrossEncoder for re-ranking
│   └── train.ipynb                    # training loop
└── eval/
    ├── metrics.py                     # hit_rate_at_k, ndcg_at_k
    ├── run_eval.ipynb                 # 5k-pair HR@10 / NDCG@10
    └── vibe_check.ipynb               # personal top-K with titles

data/
├── raw/                               # UCSD dumps (gitignored)
└── transformed/                       # all processed artifacts (gitignored)
    ├── books_with_genres.parquet
    ├── books_with_interactions.parquet  # 19 GB full
    ├── training_interactions.parquet    # 530 MB slim
    ├── book_embeddings.npy              # (1.78M, 384) float16
    ├── genre_matrix.npy                 # (1.78M, 10)
    ├── num_pages_normalized.npy         # (1.78M,)
    ├── num_pages_norm_params.json       # p1/p99/median for inference
    ├── train_user_features.npy          # (783K, 779)
    ├── train_user_features_v4.npy       # (783K, 1163) with author taste
    ├── item_features_v4.npy             # (1.78M, 779) with author embedding
    ├── author_embeddings.npy            # (386K, 384)
    ├── book_to_author_idx.npy
    ├── book_id_to_index.json
    ├── user_id_to_index.json
    ├── author_id_to_index.json
    ├── my_books.csv                     # personal export
    └── user_features.npy                # (779,) personal vector

data/checkpoints/                       # gitignored
├── two_tower_mac.pt                    # v1 — production model
├── two_tower_v2_best.pt
├── two_tower_v3_best.pt
├── two_tower_v4bce_best.pt
└── cross_encoder_v6.pt

plans/
├── book-recommender-mvp-plan.md        # this file
├── v3-overhaul-results.md
└── v3-overhaul-deferred-items.md
```

## Implementation decisions worth knowing

These are decisions that surfaced during build and are worth surfacing in the plan because they shape the rest of the system.

### Feature matrices over `nn.Embedding`

Item features are precomputed numpy arrays loaded as plain torch tensors, not `nn.Embedding` layers. Reasons: explicit memory control (5+ GB of features), no gradient buffer overhead on the static feature tables, easier to swap feature sets between experiments.

### Inverse-CDF sampling

For log-frequency negative sampling, `torch.multinomial` over 1.78M elements called per-anchor produced a 30× slowdown. Fix: precompute the cumulative distribution once, sample via `np.random.random() + np.searchsorted` — same distribution, ~100× faster.

### Sparse matmul for collaborative aggregation

Building bulk user features (rating-weighted mean over each user's liked books) uses a `(n_users, n_books)` sparse matrix multiplied by the dense `(n_books, 384)` embedding matrix. ~5 seconds for 783K users, vs 10+ minutes for a Python loop. Same pattern reused for author embeddings.

### Why temperature is learnable in `TwoTowerModel` but didn't help

L2-normalized dot products are bounded in `[-1, 1]`, which caps sigmoid in roughly `[0.27, 0.73]` — too narrow for BCE to push confident predictions. Temperature `τ` multiplies the similarity into a usable logit range. Init at `τ=10`, parameterized in log space (so Adam updates can't push it negative).

For BCE this works fine — `τ` ends up around 28 in v1.

For InfoNCE this caused divergence: the model found it easier to crank temperature than to improve embeddings. Documented in v3-overhaul-results.

### Granite embeddings over MiniLM

`ibm-granite/granite-embedding-30m-english` is what's actually used (same 384 dim as MiniLM, faster on the user's setup). Plan originally specified MiniLM; the swap was made during step 11 (embed.ipynb).

### Genre count is 10, not 25

The UCSD `book_genres_initial.json` has 10 top-level categories (not 25 as the original plan estimated). Item tower input dim is 395 (not ~410 as v1 plan said); user tower is 779 (not ~794).

## Further notes

- The UCSD Goodreads dataset requires emailing the authors for access (sites.google.com/eng.ucsd.edu/ucsdbookgraph). Plan for a few days wait time.
- Sentence-transformer embedding of the filtered book set on a T4 GPU takes 15–30 minutes. Run once via `scripts/embed.ipynb`, save to Drive/HF Hub, reuse across training runs.
- At inference time, normalization params (num_pages min/max, genre vocab ordering, author_id_to_index) must be identical to those used during training. Save them as artifacts alongside the model.
- On Colab/Cloud, set `MYBOOKREC_DATA_DIR` env var before importing the package to redirect `DATA_DIR` to wherever data lives in that environment (typically `/content/drive/MyDrive/MyBookRec/data`).
- Cross-encoder is the only model trained with BCE on independent (user, item) pairs (no in-batch). Two-tower variants use either BCE-with-sampled-negatives or InfoNCE-with-in-batch.
