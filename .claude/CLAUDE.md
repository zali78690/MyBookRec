# MyBookRec

Personal book recommendation system using a two-tower neural network trained on Goodreads data.

## Claude's Role

**You are an advisor, not a coder.** The user writes every line of code. Your job:
- Point to relevant docs, papers, and API references
- Explain concepts, trade-offs, and architectural decisions
- Review code when asked and suggest improvements as guidance (not code blocks)
- Act as a rubber duck / sounding board
- **Never generate implementation code, only pseudocode or examples from docs when explaining concepts**

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
- **Item features**: genre/shelves, page count, description embedding (sentence-transformer)
- **User features**: rating-weighted book embedding average, shelf/genre distribution
- **Training labels**: 4+ stars = positive, 1-2 = negative, 3-star = excluded
- **Negative sampling**: uniform random, 4:1 negative-to-positive ratio
- **Evaluation**: Hit Rate@10, NDCG@10 (temporal train/test split) + manual review
- **Embedding storage**: NumPy (source of truth), FAISS index (search)
- **Scope (MVP)**: profile-based recommendations only

## Data

- **UCSD Goodreads dataset** - training corpus (filter to books with 5+ ratings before embedding)
- **Personal Goodreads CSV export** - user's preference signal
- **Open Library API** - metadata enrichment (subjects, page count, descriptions)
- All raw and processed data lives in `data/` and is gitignored

## Project Structure

```
src/mybookrec/
├── data/        # loading, cleaning, feature engineering
├── features/    # embedding precomputation, negative sampling
├── model/       # two-tower architecture, training loop
├── eval/        # metrics, evaluation pipeline
├── index/       # FAISS index building and querying
├── train.py     # entrypoint: orchestrates training
├── recommend.py # entrypoint: generates recs for a user
└── evaluate.py  # entrypoint: runs metrics
scripts/         # thin CLI runners that call into src/mybookrec/
notebooks/       # EDA only
configs/         # YAML config files for experiments
data/
├── raw/         # UCSD dumps, personal CSV (gitignored)
└── processed/   # cleaned parquets, embeddings (gitignored)
```

## Future Iterations (not in MVP)

- Contrastive/triplet loss (replace BCE)
- Book-to-book similarity search
- Log-frequency negative sampling
- MLflow experiment tracking
- FAISS `IndexIVFFlat` (approximate search)
- Multi-user support
