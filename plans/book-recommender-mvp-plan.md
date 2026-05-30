# MyBookRec — Plan

## What this is

A two-tower neural recommender trained on UCSD Goodreads. Encodes users and books into a shared 128-dim space; at inference, dot-product the user embedding against all book embeddings to retrieve top-K. MVP scope: single-user, profile-based.

## Status

| Component | State |
|---|---|
| Data pipeline | ✅ |
| Item / user features (v1 + v4 author) | ✅ |
| TwoTowerModel + training loop | ✅ |
| Eval metrics (HR@K, NDCG@K) | ✅ |
| FAISS retrieval + recommendation CLI | ✅ |
| Vibe-check diagnostics | ✅ |
| InfoNCE loss + cross-encoder | ⚠️ built, didn't beat v1 in budget |
| MPNet 768-dim embeddings | ⏳ Colab compute pending |
| HF Hub artifact upload | ⏳ |

**Best model: `checkpoints/two_tower_mac.pt` (v1)** — BCE + uniform sampling, ~56k batches on MPS. HR@10=0.0100, NDCG@10=0.0043 on 5,000 test pairs (rng=0).

## Architecture

```
User features → UserTower MLP → L2-norm → user_emb (128) ┐
                                                          ├─ dot · τ → BCE
Item features → ItemTower MLP → L2-norm → item_emb (128) ┘
```

Both towers: `input → 512 → 256 → 128`, ReLU + dropout 0.1, Adam LR 1e-3. Learnable temperature `τ` (init 10, log-space) scales L2-normed dot product so BCE has usable logit range.

### Item features

| Channel | Dim | Source |
|---|---|---|
| Description embedding | 384 | `ibm-granite/granite-embedding-30m-english` on `"title. description"` |
| Genre vector | 10 | L2-norm count over fixed vocab |
| Normalized pages | 1 | Min-max with 1st/99th percentile clip, median-imputed |
| Author embedding (v4) | 384 | Mean of author's books' description embeddings |

v1 = 395-dim. v4 with author = 779-dim.

### User features (built from train split only — no leakage)

| Channel | Dim | Source |
|---|---|---|
| Like embedding | 384 | Rating-weighted mean of ≥4-star books' embeddings |
| Dislike embedding | 384 | Mean of 1-2-star books' embeddings (zero if none) |
| Genre distribution | 10 | L2-norm sum of liked books' genre vectors |
| Mean pages | 1 | Mean of liked books' normalized pages |
| Author taste (v4) | 384 | Rating-weighted mean of liked authors' embeddings |

v1 = 779-dim. v4 with author = 1163-dim.

## Data pipeline

1. Filter: English, ratings_count ≥ 5 (books), explicit ratings only, users ≥10 ratings.
2. Labels: ≥4★ positive, 1-2★ negative, 3★ excluded.
3. Temporal per-user split: 70 train / 10 val / 20 test by `date_added`.

The full `books_with_interactions.parquet` is 19 GB. For training we read a 530 MB slim version (`training_interactions.parquet`, zstd-compressed, only 4 columns).

## Training

`mybookrec.model.train` auto-detects v1 vs v4 features, loads to GPU, BCE on 1 positive + 4 sampled negatives per anchor, val HR@10 every 5k batches, early-stops on plateau. ~14 batches/sec on Apple Silicon MPS.

Negative sampling supports `uniform` and `log_freq` (inverse-CDF + `np.searchsorted`, ~100× faster than `torch.multinomial`).

## Evaluation

- `mybookrec.eval.evaluate` — HR@K + NDCG@K on 5,000 leave-one-out test pairs.
- `mybookrec.eval.vibe_check` — top-K personal recs + diagnostic checks against the known synthetic profile.

## Synthetic test profile

`my_books.csv` is repackaged from a single UCSD power user (id `096c015b…`, 293 books), not a real personal export. See `scripts/build_synthetic_library.py`.

Distribution: 227×5★ / 16×4★ / 8×3★ / 15×2★ / 27×1★. Love-it-or-hate-it rater — strong dislike signal.

**Likes**: YA fantasy with female protagonists, shoujo manga, Avatar comics, middle-grade, LDS texts.

**Dislikes**: All 4 Twilight books + box set rated 1★ (cleanest negative test case); school classics; adult dense fantasy; some hyped contemporary YA.

### Vibe-check signals

| Good signs | Red flags |
|---|---|
| Throne of Glass, Daughter of Smoke & Bone, Caraval | Twilight or paranormal romance → dislike emb broken |
| Fruits Basket, Skip Beat, Ouran (shoujo manga) | Adult literary fiction (McCarthy, Murakami) |
| Where the Mountain Meets the Moon, Ella Enchanted | Wheel of Time / Stormlight (adult male-led epic) |
| Avatar / Lumberjanes / Aru Shah | Top-10 all near-identical → popularity bias |
| Brandon Mull / Fablehaven | Books the user already rated → train-mask broke |

## v3 overhaul results (5,000-pair eval, rng=0)

| Run | Loss | Negatives | Features | HR@10 | Verdict |
|---|---|---|---|---|---|
| **v1** | BCE | uniform 4:1 | 779/395 | **0.0100** | production model |
| v2 | BCE+heavy reg | log_freq | 779/395 | 0.0066 | over-regularized |
| v3 | InfoNCE+in-batch | uniform | 779/395 | 0.0015→0 | τ diverged; slow convergence |
| v4-BCE | BCE | uniform | 1163/779 | 0.0086 | undertrained (15k vs v1's 56k batches) |
| v6 pipeline | v4-BCE retrieve + cross-encoder rerank | uniform | 1163/779 | 0.0054 | both undertrained |

Full details: [v3-overhaul-results.md](v3-overhaul-results.md).

**Bugs caught and fixed in `loss.py`**: in-batch positive collisions (when two users share a book) → mask off-diagonals; `-inf × τ` NaN gradient on temperature → mask after scaling.

**Lesson**: training-budget compounds. v1 had 11× more batches; new architectures need that same budget to validate.

## Remaining MVP work

1. Re-train v4-BCE with full ~60-min MPS budget that v1 got — most likely path to beat HR@10=0.0100.
2. HF Hub upload (model + features + FAISS index + mappings + vocab).
3. Cloud training notebook for Colab T4 fallback.

## Deferred (post-MVP)

Concrete next steps in [v3-overhaul-deferred-items.md](v3-overhaul-deferred-items.md).

| Improvement | Expected lift | Effort |
|---|---|---|
| MPNet 768 / BGE 1024 embeddings | 1.2-1.3× | 4 GPU-hours |
| Review text features | 1.2-1.5× | 30 GPU-hours |
| BISAC / hybrid taxonomy | 1.2-1.5× | 4 hours data |
| SASRec (transformer over sequence) | 1.5-2× | 1 week rewrite |
| Hard negative mining | 1.5-2× | 1 hour (needs working InfoNCE) |
| Temporal decay in user features | 1.2× | 30 min |

Skipped entirely: dislike genre vector, inverse-rating dislike weighting, book-to-book similarity, MLflow, `IndexIVFFlat`, multi-user serving, Pydantic config.

## Project structure

`mybookrec/` is the library. Every `.py` module is importable AND runnable via `python -m mybookrec.<dotted.path>`. `scripts/` contains only things that aren't naturally part of the library.

```
mybookrec/
├── __init__.py                         # ROOT_DIR, DATA_DIR (env override: MYBOOKREC_DATA_DIR)
├── io.py                               # FeatureSet registry, load_checkpoint, batch_encode,
│                                       #   build_train_exclude, sample_test_pairs
├── recommend.py                        # CLI: full inference with post-ranking filters
├── data_load/
│   ├── transform_raw.py                # CLI: raw JSON → cleaned parquets
│   └── build_training_interactions.py  # CLI: 19 GB → 530 MB slim
├── features/
│   ├── generate_vocab.py               # CLI: write genre_vocab.json
│   ├── genre_vocab.json
│   ├── build_item_features.py          # CLI
│   ├── build_user_features.py          # CLI (personal user vector)
│   ├── build_train_user_features.py    # CLI (bulk via sparse matmul)
│   ├── build_author_features.py        # CLI (v4)
│   └── training_pairs.py               # TrainingPairsDataset (importable)
├── model/
│   ├── towers.py                       # ItemTower, UserTower, TwoTowerModel
│   ├── loss.py                         # info_nce_in_batch (with collision mask)
│   ├── cross_encoder.py
│   └── train.py                        # CLI
├── eval/
│   ├── metrics.py                      # hit_rate_at_k, ndcg_at_k
│   ├── evaluate.py                     # CLI
│   └── vibe_check.py                   # CLI
└── index/
    ├── faiss_index.py
    └── build_index.py                  # CLI

scripts/                                # not part of the library
├── embed.ipynb                         # Colab GPU notebook (only kept notebook)
└── build_synthetic_library.py          # one-off

notebooks/EDA.ipynb                     # exploratory only

data/                                   # gitignored
├── raw/
└── transformed/                        # all processed artifacts

checkpoints/                            # gitignored — *.pt + *_index.faiss
```

## How to run

```bash
# Data prep (when raw UCSD files change)
.venv/bin/python -m mybookrec.data_load.transform_raw
.venv/bin/python -m mybookrec.data_load.build_training_interactions
.venv/bin/python -m mybookrec.features.generate_vocab

# Feature engineering (when embeddings or vocab change)
.venv/bin/python -m mybookrec.features.build_item_features
.venv/bin/python -m mybookrec.features.build_user_features
.venv/bin/python -m mybookrec.features.build_train_user_features
.venv/bin/python -m mybookrec.features.build_author_features          # v4

# Training (auto-detects v1 vs v4)
.venv/bin/python -m mybookrec.model.train --time-budget 5400

# Eval
.venv/bin/python -m mybookrec.eval.evaluate checkpoints/two_tower_v4bce_best.pt
.venv/bin/python -m mybookrec.eval.vibe_check checkpoints/two_tower_v4bce_best.pt

# Inference
.venv/bin/python -m mybookrec.index.build_index checkpoints/two_tower_v4bce_best.pt
.venv/bin/python -m mybookrec.recommend checkpoints/two_tower_v4bce_best.pt \
    --index checkpoints/two_tower_v4bce_best_index.faiss --top-k 20 --min-avg-rating 4.0
```

## Decisions worth remembering

- **Feature matrices, not `nn.Embedding`**: explicit memory control, no grad buffer on static tables, swappable across experiments.
- **Inverse-CDF sampling** beats `torch.multinomial` ~100× for many small per-call draws (DataLoader hot path).
- **Sparse matmul** for bulk user-feature aggregation: ~5s vs ~10 min for 783K users.
- **Learnable temperature** stays positive via log-space parameterisation. Works fine for BCE (settles ~28); diverges with InfoNCE (the model crank-cheats τ rather than improving embeddings).
- **Granite-30m embeddings**, not MiniLM (same dim, faster on this setup). Will swap to MPNet 768-dim when Colab compute returns.
- **Genre vocab has 10 categories** (the original plan guessed ~25). Item dim 395, user dim 779.
- **`MYBOOKREC_DATA_DIR` env var** redirects `DATA_DIR` on Colab/Cloud without touching code.
- **Adding a new feature set** is one entry in `mybookrec.io.FEATURE_SETS`. Everything downstream auto-detects by `item_input_dim`.

## Further notes

- UCSD Goodreads dataset requires emailing the authors (sites.google.com/eng.ucsd.edu/ucsdbookgraph). Few-day wait.
- Embedding 1.78M books on a T4 takes 15-30 min. Run once via `scripts/embed.ipynb`, save to Drive/HF Hub, reuse.
- Normalization params (`num_pages_norm_params.json`, `genre_vocab.json`, `author_id_to_index.json`) must travel with the model — load with the same values used at training.
