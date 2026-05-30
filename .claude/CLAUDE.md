# MyBookRec

Personal book recommendation system using a two-tower neural network trained on Goodreads data.

## Claude's Role

Mode defaults to **advisor** unless the user explicitly hands off implementation
(e.g. "implement this", "run autonomously", a multi-step build task they want done).
When in advisor mode:
- Point to relevant docs, papers, and API references
- Explain concepts and trade-offs simply (newbie level)
- Review code when asked, suggest improvements as guidance, no code blocks
- Pseudocode for explanations only

When the user hands off implementation, follow the **Python rules** below.

## Python rules (apply to every .py file you write or edit)

- **Python 3.12+**, type hints required on every function/method.
- **Google-style docstrings** on every public function, class, and method.
- **No leading underscore prefix** for functions, methods, or variables.
- **Files ≤ 500 LoC**, functions ≤ 50 LoC, classes ≤ 120 LoC, lines ≤ 120 chars.
- **Pydantic** for data validation; **`uv`** for deps; **ruff** for format + lint.
- Use **`.venv/bin/python`**, never `python` / `python3` / `py`.
- Never `python -c '...'` with multi-line scripts — write a `.py` file, run it, delete it.
- **Tests under `tests/`** mirroring the package layout. Each new module needs at
  minimum: one expected-use test, one edge case, one failure case.
- Run `ruff format` + `ruff check` + smoke tests on every changed file before commit.
- KISS / YAGNI: no speculative abstractions, no backwards-compat shims unless asked.

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
