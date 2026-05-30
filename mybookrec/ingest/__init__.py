"""Book ingestion pipeline (medallion: bronze → silver → gold).

- `bronze`: raw, immutable API responses dumped to JSONL under data/bronze/<source>/<date>/.
- `silver`: cleaned, schema-aligned parquets under data/silver/.
- `gold`: feature-ready vectors + FAISS-ready embeddings under data/gold/.

Sources are pluggable. Each adapter is a thin module:
1. `fetch_*` — paginated query against the source API, writes raw JSONL.
2. `bronze_to_silver_*` — Pydantic-validated normalization to the shared silver schema.

Cross-source merging happens in `to_silver.merge_sources` (dedupe by ISBN-13 then title+author).
"""
