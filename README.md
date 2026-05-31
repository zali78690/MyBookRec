# MyBookRec

Personal book recommender: a two-tower neural network trained on UCSD Goodreads, served
over FastAPI with FAISS retrieval, and continuously enrichable from free book APIs
(Open Library + Google Books) through a medallion-style ingestion pipeline.

See [plans/book-recommender-mvp-plan.md](plans/book-recommender-mvp-plan.md) for the
full architecture, data lineage, and current status.

## Layout

```
mybookrec/
├── settings.py            Pydantic Settings — single source of truth for config
├── data_load/             Raw → cleaned parquets (UCSD)
├── features/              Item / user / author feature builders
├── model/                 TwoTowerModel, training, loss, cross-encoder
├── eval/                  HR@K, NDCG@K, vibe-check
├── index/                 FAISS index build + query
├── ingest/                Bronze (raw API) → silver (cleaned) → gold (features) → FAISS refresh
├── serve/                 FastAPI app: POST /recommend, GET /healthz
└── recommend.py           CLI for offline top-K with post-rank filters

tests/                     Pytest suite (mirrors the package layout)
data/
├── raw/                                  Raw source files (DVC-tracked)
│   ├── ucsd/                             UCSD Goodreads scrape (books, interactions, genres)
│   ├── openlibrary/                      OL bulk dumps (works dump)
│   └── personal/                         Personal Goodreads CSV export
├── transformed/                          Cleaned parquets + .npy artifacts (DVC-tracked)
│   ├── shared/                           Model-independent (id maps, genre matrix,
│   │                                       books_with_genres, training_interactions,
│   │                                       isbn index, my_books, embedding_input)
│   ├── v1_minilm/                        MiniLM-384 embeddings + downstream features
│   │                                       (book_embeddings, author_embeddings,
│   │                                       item_features, user_features, ...)
│   └── v2_mxbai/                         mxbai-512 — created when MPNet/mxbai pass completes
├── bronze/                               Immutable raw API responses (date-partitioned JSONL)
│   ├── openlibrary/<date>/               Per-query Open Library Search fetches
│   ├── openlibrary_dump/<tag>/           OL bulk-dump shards
│   └── google_books/<date>/              Google Books fetches
├── silver/                               Cleaned schema-aligned book parquets
└── gold/                                 Embeddings + feature vectors ready for FAISS
checkpoints/                              .pt model + .faiss index artifacts
```

## Setup

```bash
uv sync --all-groups
cp .env.example .env  # fill in GOOGLE_BOOKS_API_KEY (free)
```

## Quick tasks

```bash
# --- training / eval (UCSD batch) ---
.venv/bin/python -m mybookrec.model.train --time-budget 5400
.venv/bin/python -m mybookrec.eval.evaluate checkpoints/two_tower_mac.pt
.venv/bin/python -m mybookrec.eval.vibe_check checkpoints/two_tower_mac.pt

# --- ingest fresh books (medallion) ---
.venv/bin/python -m mybookrec.ingest.cli fetch  --query "brandon sanderson" --source both
.venv/bin/python -m mybookrec.ingest.cli silver
.venv/bin/python -m mybookrec.ingest.cli gold
.venv/bin/python -m mybookrec.ingest.cli refresh --index checkpoints/two_tower_mac_index.faiss

# --- serve ---
.venv/bin/python -m mybookrec.serve                           # local
docker compose up --build                                     # containerised
curl -s http://localhost:8000/healthz | jq

# --- tests + lint ---
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff format mybookrec tests
.venv/bin/ruff check mybookrec tests

# --- data versioning (DVC) ---
.venv/bin/dvc repro          # rebuild any out-of-date stages
.venv/bin/dvc push           # push to the configured local remote
```

## Real-time prediction

The two-tower split is what makes serving real-time:

```
POST /recommend
  body: { ratings: [{book_id, rating}, ...], top_k, ebook_only?, min_avg_rating? }
  steps:
    1. compute user features from supplied ratings  (~20 ms)
    2. encode_user through the UserTower            (~5 ms)
    3. FAISS top-K*oversample over 1.78M items      (~30 ms)
    4. apply post-rank filters                      (~10 ms)
    p99 latency target: <100 ms
```

User features are rebuilt on every request — no per-user state — so a new user's first
request is the same as their thousandth. New books enter via the ingest pipeline and
are appended to the FAISS index in place by `ingest.cli refresh`.

## Config

Every runtime knob lives in [mybookrec/settings.py](mybookrec/settings.py). Settings are
loaded from environment variables, then `.env`, then defaults — in that order. See
[.env.example](.env.example) for the available vars.

## Conventions

See [.claude/CLAUDE.md](.claude/CLAUDE.md). Highlights:
- Python 3.12+, type hints everywhere, Google-style docstrings.
- `uv` for deps; `ruff format` + `ruff check` before commit.
- No leading underscores on functions / methods / variables.
- Tests under `tests/` mirroring the package layout — expected, edge, failure case minimum.
