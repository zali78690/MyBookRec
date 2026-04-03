# MyBookRec MVP - Product Requirements Document

## Problem Statement

Goodreads recommendations are generic and don't account for individual taste nuance. As someone who actively rates and shelves books, I want a recommendation system that learns from my specific patterns — what genres I gravitate toward, what book lengths I prefer, what content resonates — and surfaces books I'd actually enjoy, not just popular titles in broad categories.

Goodreads uses collaborative filtering — "users who rated the same books as you also liked these other books." It finds similar users and surfaces what they read.

This system is content + behavior hybrid:        
  - It understands what a book is about (via description embeddings, genre vectors)
  - It learns your taste profile from your behavior
  - It can recommend books with zero ratings from  similar users — as long as the content matches your learned taste

Concrete difference: if a niche book was just published and no one similar to you has rated it, Goodreads can't recommend it. Your model can,because it reasons about content, not just    co-occurrence.

## Solution

A two-tower neural network recommendation system trained on the UCSD Goodreads dataset that generates personalized book recommendations based on a user's rating history. The MVP is profile-based: given a user's reading history, produce a ranked list of books they're likely to enjoy.

The system encodes users and books into a shared embedding space where proximity indicates relevance. At inference time, the user's embedding is compared against all book embeddings via FAISS to retrieve top-K recommendations.

## User Stories

1. As a reader, I want to load my Goodreads export CSV so that the system knows what I've read and how I rated it.
2. As a reader, I want to receive a ranked list of book recommendations based on my rating history so that I can find books I'll enjoy.
3. As a reader, I want to see why a book was recommended (e.g., similar to books I rated highly) so that I can trust the recommendations.
4. As a reader, I want to exclude books I've already read from recommendations so that results are actionable.
5. As a reader, I want to control how many recommendations I receive (top-K) so that I can browse at my preferred depth.
6. As a developer, I want to run the full pipeline (data → features → train → evaluate → recommend) from CLI scripts so that the workflow is reproducible.
7. As a developer, I want to configure hyperparameters via YAML files so that I can track and compare experiments.
8. As a developer, I want offline evaluation metrics (Hit Rate@10, NDCG@10) so that I can measure whether model changes are improvements.
9. As a developer, I want to precompute and cache book embeddings so that I don't re-run the sentence-transformer on every training run.
10. As a developer, I want temporal train/validation/test splits so that evaluation reflects real-world prediction (no future leakage) and I can tune hyperparameters without contaminating the test set.
11. As a developer, I want to save and load trained model checkpoints so that I don't retrain from scratch every session.
12. As a developer, I want to save and load the FAISS index so that recommendation serving doesn't require rebuilding the index.
13. As a developer, I want the data pipeline to filter low-signal records (books with <5 ratings, users with <10 ratings) so that training is efficient and meaningful.
14. As a future maintainer, I want the project structured so that other users can plug in their own Goodreads export and get recommendations without modifying code.
15. As a developer, I want trained artifacts (model, index, embeddings, config) stored on Hugging Face Hub so that they are versioned, shareable, and don't require retraining to use.
16. As a developer, I want a cloud training notebook that handles Kaggle/Colab environment setup and calls my existing scripts so that I can train on free-tier GPUs without modifying my codebase.

## Implementation Decisions

### Data Pipeline (`mybookrec.data`)
- UCSD Goodreads dataset as training corpus, loaded from JSON/CSV dumps
- Personal Goodreads CSV export as the query user's preference signal
- All raw data stored in `data/raw/`, processed data in `data/processed/` — both gitignored
- Polars for all data processing; output as parquet files
- **Language filter**: drop all books where `language_code` is not in `{"eng", "en-US", "en-GB", ""}` — applied before any feature engineering. Empty string is kept because many English books in the UCSD dataset have missing language codes.
- **Rating filter**: drop all interactions where `rating == 0` (shelved but not rated — no taste signal). Only explicit 1–5 star ratings are used.
- **Quality filter**: keep only books with 5+ ratings and users with 10+ explicit ratings
- **Training labels**: `rating >= 4` → positive (1), `rating in {1, 2}` → negative (0), `rating == 3` → excluded (ambiguous)

### Feature Engineering (`mybookrec.features`)

#### Item Features
- **Description embedding**: concatenate `"{title}. {description}"` then embed with `all-MiniLM-L6-v2` (384-dim). Precomputed once, stored as NumPy `.npy` with a book_id ↔ row index mapping.
- **Genre vector**: built from `book_genres_initial.json`. Find all unique genre keys across the corpus (~25 categories), build a count-weighted vector per book, L2-normalize. Fixed genre ordering stored as a vocab file.
- **Page count**: `num_pages` min-max normalized to [0, 1] using 1st–99th percentile range to avoid outlier skew. Verify empty strings in EDA and impute with median if present.

Item tower input dimensionality: ~410 (384 description + ~25 genre + 1 pages)

#### User Features (built from personal Goodreads CSV)
- **Like embedding** (384-dim): rating-weighted average of description embeddings for all books rated 4+ stars. Weight = rating value (5-star counts more than 4-star).
- **Dislike embedding** (384-dim): simple average of description embeddings for all books rated 1–2 stars. If fewer than ~10 disliked books exist, this vector will be noisy — verify in EDA.
- **Genre distribution** (~25-dim): aggregated genre vector from all 4+ star rated books, L2-normalized. Uses same genre vocab as item features.
- **Mean page preference** (1-dim): mean of normalized `num_pages` across all 4+ star rated books.

User tower input dimensionality: ~794 (384 like + 384 dislike + ~25 genre + 1 pages)

#### Training Pairs
- Negative sampling: uniform random, 4 negatives per 1 positive, re-sampled each epoch
- Temporal split per user ordered by rating date: train (70%), validation (10%), test (20%)
- Validation set used for early stopping and hyperparameter tuning
- Test set held out completely, only used for final model evaluation

### Model (`mybookrec.model`)
- Two-tower architecture: separate `UserTower` and `ItemTower` (both `nn.Module`)
- Each tower maps its input features through an MLP to a shared embedding space
- Similarity via dot product between user and item embeddings
- Loss: `BCEWithLogitsLoss`
- Raw PyTorch training loop (no Lightning)
- Device-agnostic: auto-detects CUDA, falls back to CPU
- Optimizer: Adam
- Early stopping based on validation Hit Rate@10
- Model checkpoint save/load via `torch.save` / `torch.load`

### Post-Ranking Filters
Applied after FAISS retrieval, before presenting results to the user. Configurable via YAML:
- `min_avg_rating` (float, default 3.5): exclude books below this global average rating
- `ebook_only` (bool, default false): restrict results to books where `is_ebook == true`
- Already-read books are always excluded (matched against personal Goodreads CSV)

These are format/quality constraints, not taste signals — keeping them out of the model prevents popularity bias and format conflation.

### Evaluation (`mybookrec.eval`)
- Hit Rate@10: was the held-out book in the top 10 predictions?
- NDCG@10: accounts for ranking position of correct predictions
- Validation metrics computed each epoch to guide training decisions
- Test metrics computed once on final model only
- Personal vibe check: generate top-20 for the user's own profile and manually review

### Index (`mybookrec.index`)
- FAISS `IndexFlatIP` (exact inner product search)
- Built from item tower output embeddings (not raw sentence-transformer embeddings)
- Serialized via `faiss.write_index` / `faiss.read_index`
- Query: user embedding → top-K book IDs with similarity scores

### Configuration (`mybookrec.config`)
- Pydantic `BaseSettings` class covering all paths and hyperparameters
- Loaded from YAML files in `configs/`
- Environment variable overrides supported via `.env`
- Post-ranking filter params (`min_avg_rating`, `ebook_only`) live here

### Artifact Storage (Hugging Face Hub)
- Trained model checkpoint (`.pt`)
- FAISS index (`.faiss`)
- Precomputed embedding matrix (`.npy`) + book_id ↔ row index mapping
- Genre vocab file (fixed ordering for genre vectors)
- Config YAML used for the training run (reproducibility)
- Upload/download via `huggingface_hub` Python library

### Project Structure
- `src/mybookrec/` — all library code in subpackages (data, features, model, eval, index, config)
- `src/mybookrec/train.py`, `evaluate.py`, `recommend.py` — entrypoints within the package
- `scripts/` — thin CLI runners: `train.py`, `evaluate.py`, `recommend.py`, `embed.py`
- `notebooks/eda/` — Jupyter notebooks for exploratory data analysis
- `notebooks/cloud_train.ipynb` — cloud environment setup + calls scripts for GPU training
- `configs/` — YAML experiment configs

### Compute Strategy
- Local CPU for development, EDA, and debugging
- Free-tier GPU (Kaggle Notebooks or Google Colab) for embedding precomputation and model training
- `cloud_train.ipynb` handles environment setup (install deps, download data from source, run scripts, upload artifacts to HF Hub)

## Testing Decisions

A good test verifies **external behavior and data contracts**, not internal implementation.

### Modules with tests:

**`mybookrec.features`**
- Negative sampling produces correct 4:1 ratio and no overlap with positives
- Embedding matrix has expected shape (num_books × 384)
- Temporal split: all validation dates > all train dates, all test dates > all validation dates, per user
- User like/dislike vectors have correct dimensionality (384)
- Genre vector has correct dimensionality and sums to 1.0 after L2 normalization
- `rating == 0` and `rating == 3` interactions are excluded from training pairs
- Language filter removes non-English books

**`mybookrec.model`**
- Forward pass produces correct output shapes for given input dimensions
- Training loop reduces loss over a small number of batches (smoke test)
- User and item tower outputs are the same embedding dimension
- Model save/load round-trips correctly

**`mybookrec.eval`**
- Hit Rate@K with synthetic data where answer is known
- NDCG@K with synthetic data where answer is known
- Edge cases: user with no test items, K larger than candidate set

### Modules without dedicated tests:
- `mybookrec.data` — integration-tested through the pipeline entrypoints
- `mybookrec.index` — thin FAISS wrapper, tested via end-to-end recommendation flow
- `mybookrec.config` — Pydantic validates at instantiation; no custom logic to test

## Out of Scope (MVP)

- **Open Library API enrichment** — all MVP item features (num_pages, description, is_ebook, language_code, genres) are available in the UCSD dataset directly. No external API needed.
- **Author features** — description embeddings partially capture authorial style. A proper implementation (author ID embeddings or author-aggregated embeddings) needs enough per-author interactions to train well. Deferred to v2.
- **Book-to-book similarity search** — trivially available once item tower is trained, but not part of MVP
- **Contrastive / triplet loss** — future upgrade from BCE
- **Log-frequency negative sampling** — future upgrade from uniform random
- **MLflow experiment tracking** — config structure supports it, deferred
- **FAISS approximate search (IndexIVFFlat)** — unnecessary at current scale
- **Multi-user serving / web interface** — MVP is CLI-only for a single user
- **Real-time Goodreads sync** — manual CSV export only
- **Review text as a user feature** — user primarily rates and shelves

## Future Scope Notes

- **Dislike embedding weighting**: for MVP, dislike embedding is a simple average of 1–2 star book embeddings. A v2 improvement is inverse-rating weighting (1-star gets more weight than 2-star).
- **Author embeddings**: two viable approaches for v2 — (A) author ID lookup table trained end-to-end, (B) precomputed per-author embedding as mean of their books' description embeddings. Option B avoids cold-start for obscure authors.
- **Explicit dislike genre signal**: currently genre distribution is built from positives only. A v2 "dislike genre vector" (from 1–2 star books) could be added to the user tower as a separate input.
- **Temporal decay on like embedding**: for MVP, the like embedding treats all 4+ star ratings equally. A v2 improvement is multiplying each book's weight by `exp(-λ * days_since_rating)` so recent ratings count more than old ones. `λ` would be a config hyperparameter (λ = 0 recovers MVP behavior). Requires verifying `date_added` coverage in EDA first — if the Goodreads CSV has sparse or unparseable dates, this signal degrades. Same decay could optionally be applied to the genre distribution.

---

## Step-by-Step Build Guide

This is a rough ordered guide from EDA to a working local MVP. Steps within a phase can overlap; phases are sequential.

### Phase 0: EDA (`notebooks/EDA.ipynb`)

1. **Inspect `books_df`**
   - Check `num_pages`: count empty strings (not just nulls — it's a str column)
   - Check `language_code`: distribution of values, how many empty strings, how many non-English
   - Check `description`: coverage (how many empty?), typical length
   - Check `is_ebook`: value distribution

2. **Inspect `interactions_df`**
   - Rating distribution: how many 0s, 1s, 2s, 3s, 4s, 5s
   - After dropping 0-rated rows: how many users remain with 10+ ratings?
   - Temporal coverage: date range of ratings, are dates parseable?

3. **Inspect `book_genres_initial.json`**
   - How many unique genre keys exist?
   - What fraction of books have genre data?
   - Distribution of genre counts per book

4. **Inspect personal Goodreads CSV**
   - How many books rated? Rating distribution?
   - How many 1–2 star books (dislike signal quality)?
   - How many 4+ star books (like signal quality)?
   - Do book titles/IDs match UCSD dataset?

---

### Phase 1: Data Pipeline (`mybookrec/data/`)

5. **Load and filter interactions**
   - Load `goodreads_interactions_dedup.json.gz`
   - Drop `rating == 0`
   - Keep users with 10+ remaining ratings

6. **Load and filter books**
   - Load `goodreads_books.json.gz`
   - Filter `language_code` to English (`{"eng", "en-US", "en-GB", ""}`)
   - Filter to books with 5+ ratings (join on `ratings_count` or compute from filtered interactions)
   - Handle `num_pages` empty strings: cast to int, impute missing with median

7. **Load genres**
   - Load `goodreads_book_genres_initial.json.gz`
   - Join onto books on `book_id`

8. **Join and save**
   - Inner join books + interactions + genres
   - Save as parquet: `data/processed/books.parquet`, `data/processed/interactions.parquet`

9. **Temporal train/val/test split**
   - Per user, sort interactions by `date_added`
   - Split 70% train / 10% val / 20% test
   - Save as separate parquets

---

### Phase 2: Feature Engineering (`mybookrec/features/`)

10. **Build genre vocab**
    - Collect all unique genre keys across the corpus
    - Save as an ordered list to `data/processed/genre_vocab.json` — this is the fixed mapping for all genre vectors

11. **Precompute book embeddings** (`scripts/embed.py`)
    - For each book: `text = f"{title}. {description}"`
    - Run through `all-MiniLM-L6-v2` in batches
    - Save as `data/processed/book_embeddings.npy`
    - Save `book_id → row index` mapping as `data/processed/book_id_to_idx.json`
    - *Run on GPU (Kaggle/Colab); upload to HF Hub and reuse*

12. **Build item feature matrix**
    - Genre vector: for each book, build count vector over genre vocab, L2-normalize
    - Pages: min-max normalize using 1st–99th percentile of the corpus
    - Save normalization params (min, max) to config/artifacts for inference reuse

13. **Build user features from personal Goodreads CSV**
    - Match your rated books to UCSD `book_id`s
    - Like embedding: rating-weighted mean of embeddings for 4+ star books
    - Dislike embedding: simple mean of embeddings for 1–2 star books
    - Genre distribution: sum genre vectors of 4+ star books, L2-normalize
    - Mean pages: mean of normalized `num_pages` for 4+ star books

14. **Build training pairs**
    - For each user in train split: collect positives (4+) and sample 4× negatives (uniform random from unrated books)
    - Re-sample negatives each epoch during training (don't precompute a static negative set)
    - Exclude 3-star and 0-star rated books from negative pool for that user

---

### Phase 3: Model (`mybookrec/model/`)

15. **Define `ItemTower`**
    - Input: `[genre_vector (~25) | normalized_pages (1) | description_embedding (384)]` → ~410-dim
    - Architecture: MLP → shared embedding dim (e.g. 64 or 128 — tune via config)
    - L2-normalize output

16. **Define `UserTower`**
    - Input: `[like_embedding (384) | dislike_embedding (384) | genre_dist (~25) | mean_pages (1)]` → ~794-dim
    - Architecture: MLP → same shared embedding dim
    - L2-normalize output

17. **Training loop**
    - Similarity: dot product of user and item embeddings
    - Loss: `BCEWithLogitsLoss`
    - Optimizer: Adam
    - Resample negatives each epoch
    - Compute val Hit Rate@10 after each epoch
    - Early stopping: save checkpoint when val Hit Rate@10 improves

---

### Phase 4: Evaluation (`mybookrec/eval/`)

18. **Write metric functions**
    - `hit_rate_at_k(user_embedding, test_book_ids, all_item_embeddings, k)`
    - `ndcg_at_k(...)` — same signature
    - Test with synthetic data first (unit tests)

19. **Run evaluation on test split**
    - For each user in test split: compute their user embedding from train interactions, retrieve top-K, check against held-out test books
    - Report mean Hit Rate@10 and NDCG@10

20. **Personal vibe check**
    - Generate top-20 recs for your own Goodreads CSV
    - Manually review: do these look like books you'd actually read?

---

### Phase 5: FAISS Index + Recommendations (`mybookrec/index/`, `mybookrec/recommend.py`)

21. **Build FAISS index**
    - Run all books through trained `ItemTower` to get item embeddings
    - Build `IndexFlatIP` over all item embeddings
    - Save: `faiss.write_index(index, "data/processed/book_index.faiss")`

22. **Recommendation pipeline**
    - Load user's Goodreads CSV → compute user features → run through `UserTower` → get user embedding
    - Query FAISS index for top-K × 3 (oversample to allow for post-filtering)
    - Apply post-ranking filters: `min_avg_rating`, `ebook_only`
    - Exclude already-read books
    - Return top-K with titles and similarity scores

---

### Phase 6: Artifact Storage

23. **Upload to Hugging Face Hub**
    - `book_embeddings.npy` + `book_id_to_idx.json`
    - `book_index.faiss`
    - Model checkpoint (`.pt`)
    - Genre vocab (`genre_vocab.json`)
    - Normalization params
    - Config YAML

24. **Cloud training notebook** (`notebooks/cloud_train.ipynb`)
    - Environment setup (install deps via uv or pip)
    - Download UCSD data
    - Run `scripts/embed.py` on GPU
    - Run `scripts/train.py`
    - Upload artifacts to HF Hub

---

### Later Scope (post-MVP, in rough priority order)

- Author embeddings (v2 item feature)
- Dislike genre vector in user tower
- Inverse-rating weighting for dislike embedding
- Contrastive/triplet loss (replace BCE)
- Log-frequency negative sampling
- MLflow experiment tracking
- FAISS `IndexIVFFlat` (approximate search at scale)
- Book-to-book similarity search
- Multi-user support

---

## Further Notes

- The UCSD Goodreads dataset requires emailing the authors for access (sites.google.com/eng.ucsd.edu/ucsdbookgraph). Plan for a few days wait time.
- Sentence-transformer embedding of the filtered book set on a T4 GPU takes roughly 15–30 minutes. Run once via `scripts/embed.py`, upload to HF Hub, reuse across training runs.
- `num_pages` is a `str` column in the raw data — pandas `.info()` showing "non-null" does not mean no empty strings. Verify in EDA before assuming it's clean.
- At inference time, normalization params (num_pages min/max, genre vocab ordering) must be identical to those used during training. Save them as artifacts alongside the model.
- The architecture is designed so that upgrading from BCE to contrastive loss requires changing only the loss function and training pair generation — tower architectures stay the same.
