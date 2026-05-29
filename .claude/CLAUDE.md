# MyBookRec

Personal book recommendation system using a two-tower neural network trained on Goodreads data.

## Claude's Role

**You are an advisor, not a coder.** The user writes every line of code. Your job:
- Point to relevant docs, papers, and API references
- Explain concepts, trade-offs, and architectural decisions as simply as possible, imagine this is for a newbie
- Review code when asked and suggest improvements as guidance (not code blocks)
- Act as a rubber duck / sounding board
- **Never generate implementation code, only pseudocode or examples from docs when explaining concepts unless user overrides**

## Tech Stack

- **Python 3.12+** with `uv` for dependency management
- **PyTorch** (raw, no Lightning) - two-tower recommendation model
- **Polars** - data processing and feature engineering
- **sentence-transformers** (`all-MiniLM-L6-v2`) - precomputed book text embeddings, CPU only
- **FAISS** (`faiss-cpu`, `IndexFlatIP`) - nearest-neighbor search over book embeddings
- **Pydantic** (`BaseSettings`) + YAML - configuration and hyperparameters
- **Jupyter notebooks** - EDA only, all logic in `.py` modules

## Architecture Decisions

- **Two-tower model**: separate user and item towers producing embeddings, trained with BCE loss
- **Item features (~410-dim)**: genre vector (book_genres_initial.json, count-weighted, L2-norm, ~25-dim) + normalized page count (1-dim) + description embedding (title prepended, all-MiniLM-L6-v2, 384-dim)
- **User features (~794-dim)**: like embedding (rating-weighted avg of 4+ star book embeddings, 384-dim) + dislike embedding (avg of 1-2 star book embeddings, 384-dim) + genre distribution (from 4+ star books, ~25-dim) + mean page preference (1-dim)
- **Training labels**: 4+ stars = positive, 1-2 = negative, 3-star = excluded, 0 = excluded (no taste signal)
- **Negative sampling**: uniform random, 4:1 negative-to-positive ratio, re-sampled each epoch
- **Post-ranking filters**: min_avg_rating (quality gate) and ebook_only (format filter) — config-driven, not model features
- **Evaluation**: Hit Rate@10, NDCG@10 (temporal train/test split) + manual review
- **Embedding storage**: NumPy (source of truth), FAISS index (search)
- **Scope (MVP)**: profile-based recommendations only

## Data

- **UCSD Goodreads dataset** - training corpus; filter to English-only books, books with 5+ ratings, users with 10+ ratings
- **Personal Goodreads CSV export** - user's preference signal
- All raw and processed data lives in `data/` and is gitignored
- No external API enrichment — all MVP features are available in the UCSD dataset

## Project Structure

```
mybookrec/
├── data/        # loading, cleaning, filtering (language, ratings, quality)
├── features/    # embedding precomputation, genre vectors, user feature construction, negative sampling
├── model/       # two-tower architecture (UserTower, ItemTower), training loop
├── eval/        # metrics (Hit Rate@K, NDCG@K), evaluation pipeline
├── index/       # FAISS index building and querying
├── config.py    # Pydantic BaseSettings, loaded from configs/ YAML
├── train.py     # entrypoint: orchestrates training
├── recommend.py # entrypoint: generates recs for a user
└── evaluate.py  # entrypoint: runs metrics
scripts/                  # thin CLI runners that call into src/mybookrec/
├── embed.py              # precompute book embeddings (run on GPU, upload to HF Hub)
├── train.py
├── evaluate.py
└── recommend.py
notebooks/
├── EDA.ipynb             # exploratory data analysis only
└── cloud_train.ipynb     # GPU training on Kaggle/Colab
configs/                  # YAML experiment configs (hyperparams, post-ranking filter thresholds)
data/
├── raw/                  # UCSD dumps, personal CSV (gitignored)
└── processed/            # cleaned parquets, embeddings, genre vocab, FAISS index (gitignored)
plans/                    # PRD and step-by-step build guide
```

## Future Iterations (not in MVP)

- Author features (author ID embeddings or author-aggregated embeddings)
- Dislike genre vector in user tower
- Inverse-rating weighting for dislike embedding
- Contrastive/triplet loss (replace BCE)
- Book-to-book similarity search
- Log-frequency negative sampling
- MLflow experiment tracking
- FAISS `IndexIVFFlat` (approximate search)
- Multi-user support
