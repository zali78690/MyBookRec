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
