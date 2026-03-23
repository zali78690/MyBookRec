# MyBookRec MVP - Product Requirements Document

## Problem Statement

Goodreads recommendations are generic and don't account for individual taste nuance. As someone who actively rates and shelves books, I want a recommendation system that learns from my specific patterns — what genres I gravitate toward, what book lengths I prefer, what content resonates — and surfaces books I'd actually enjoy, not just popular titles in broad categories.

## Solution

A two-tower neural network recommendation system trained on the UCSD Goodreads dataset, enriched with Open Library metadata, that generates personalized book recommendations based on a user's rating and shelving history. The MVP is profile-based: given a user's reading history, produce a ranked list of books they're likely to enjoy.

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
13. As a developer, I want the data pipeline to filter low-signal records (books with <5 ratings) so that training is efficient and meaningful.
14. As a developer, I want to enrich book metadata from Open Library (subjects, page count) so that the item tower has richer features.
15. As a future maintainer, I want the project structured so that other users can plug in their own Goodreads export and get recommendations without modifying code.
16. As a developer, I want trained artifacts (model, index, embeddings, config) stored on Hugging Face Hub so that they are versioned, shareable, and don't require retraining to use.
17. As a developer, I want a cloud training notebook that handles Kaggle/Colab environment setup and calls my existing scripts so that I can train on free-tier GPUs without modifying my codebase.

## Implementation Decisions

### Data Pipeline (`mybookrec.data`)
- UCSD Goodreads dataset as training corpus, loaded from JSON/CSV dumps
- Personal Goodreads CSV export as the query user's preference signal
- Open Library API for metadata enrichment (subjects/genres, page count, descriptions)
- All raw data stored in `data/raw/`, processed data in `data/processed/` — both gitignored
- Polars for all data processing; output as parquet files
- Filter to books with 5+ ratings and users with 10+ ratings before any feature engineering

### Feature Engineering (`mybookrec.features`)
- Book description embeddings via `all-MiniLM-L6-v2` (sentence-transformers), precomputed once and stored as NumPy `.npy` files with a separate book_id ↔ row index mapping
- Embedding precomputation is a separate pipeline step (`scripts/embed.py`), run once on GPU and cached to Hugging Face Hub
- Item features: genre/shelf tags, page count, description embedding (384-dim)
- User features: rating-weighted average of book embeddings for books they've rated, shelf/genre distribution vector
- Training pairs: binary labels (4+ stars = 1, 1-2 stars = 0, 3 stars excluded)
- Negative sampling: uniform random, 4 negatives per 1 positive, re-sampled each epoch
- Temporal split per user ordered by rating date: train (70%), validation (10%), test (20%)
- Validation set used for early stopping and hyperparameter tuning during training
- Test set held out completely, only used for final model evaluation

### Model (`mybookrec.model`)
- Two-tower architecture: separate `UserTower` and `ItemTower` (both `nn.Module`)
- Each tower maps its input features to a shared embedding space
- Similarity via dot product between user and item embeddings
- Loss: `BCEWithLogitsLoss`
- Raw PyTorch training loop (no Lightning)
- Device-agnostic: auto-detects CUDA and uses GPU when available, falls back to CPU
- Optimizer: Adam (standard starting point)
- Early stopping based on validation Hit Rate@10
- Model checkpoint save/load via `torch.save` / `torch.load`

### Evaluation (`mybookrec.eval`)
- Hit Rate@10: was the held-out book in the top 10 predictions?
- NDCG@10: accounts for ranking position of correct predictions
- Validation metrics computed each epoch to guide training decisions
- Test metrics computed once on final model only
- Personal vibe check: generate top-20 for the user's own profile and manually review

### Index (`mybookrec.index`)
- FAISS `IndexFlatIP` (exact inner product search, no approximation)
- Built from item tower output embeddings (not raw sentence-transformer embeddings — the trained item tower refines them)
- Serialized via `faiss.write_index` / `faiss.read_index`
- Query: user embedding → top-K book IDs with similarity scores

### Configuration (`mybookrec.config`)
- Pydantic `BaseSettings` class covering all paths and hyperparameters
- Loaded from YAML files in `configs/`
- Environment variable overrides supported via `.env`

### Artifact Storage (Hugging Face Hub)
- Trained model checkpoint (`.pt`)
- FAISS index (`.faiss`)
- Precomputed embedding matrix (`.npy`) + book_id ↔ row index mapping
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
- `cloud_train.ipynb` handles environment setup (install deps, download data/embeddings from HF Hub, run scripts, upload artifacts back to HF Hub)

## Testing Decisions

A good test for this project verifies **external behavior and data contracts**, not internal implementation. Tests should answer: "given this input, does the module produce output with the correct shape, type, and statistical properties?"

### Modules with tests:

**`mybookrec.features`**
- Negative sampling produces correct ratio (4:1) and no overlap with positives
- Embedding matrix has expected shape (num_books × 384)
- Temporal split ordering: all validation dates > all train dates, all test dates > all validation dates, per user
- User feature vectors have correct dimensionality
- 3-star ratings are excluded from training pairs

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

## Out of Scope

- **Book-to-book similarity search** — the FAISS index supports this trivially once the item tower is trained, but it's not part of MVP
- **Contrastive / triplet loss** — future upgrade from BCE
- **Log-frequency negative sampling** — future upgrade from uniform random
- **MLflow experiment tracking** — config structure supports it, but integration is deferred
- **FAISS approximate search (IndexIVFFlat)** — unnecessary at current scale
- **Multi-user serving / web interface** — MVP is CLI-only for a single user
- **Real-time Goodreads sync** — manual CSV export only
- **Review text as a user feature** — user primarily rates and shelves, doesn't write reviews

## Further Notes

- The UCSD Goodreads dataset requires emailing the authors for access (sites.google.com/eng.ucsd.edu/ucsdbookgraph). Plan for a few days wait time.
- Sentence-transformer embedding of the filtered book set (~500K-800K books) on a T4 GPU takes roughly 15-30 minutes. Run once via `scripts/embed.py`, upload to HF Hub, and reuse across training runs.
- The architecture is designed so that upgrading from BCE to contrastive loss requires changing only the loss function and training pair generation — the tower architectures remain the same.
- When adding multi-user support later, each new user only needs their Goodreads CSV. The trained model and FAISS index are reusable — only the user embedding computation runs per-user.
