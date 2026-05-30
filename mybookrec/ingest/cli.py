"""Unified CLI for the ingestion pipeline.

Sub-commands:

    fetch     run one or more source fetchers, writing bronze JSONL
    silver    parse all bronze files, dedupe across sources, write silver parquet
    gold      embed silver descriptions and build feature vectors (writes gold parquet + npy)
    refresh   incrementally add new gold items to the live FAISS index

Usage:
    .venv/bin/python -m mybookrec.ingest.cli fetch --query "mistborn" --source both
    .venv/bin/python -m mybookrec.ingest.cli silver
    .venv/bin/python -m mybookrec.ingest.cli gold
    .venv/bin/python -m mybookrec.ingest.cli refresh --index checkpoints/two_tower_mac_index.faiss
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from mybookrec.ingest import google_books, openlibrary, to_silver
from mybookrec.settings import get_settings


def slugify(text: str) -> str:
    """Lowercase + dash-collapse a string into a filesystem-safe slug.

    Args:
        text: Free text (e.g. a search query).

    Returns:
        Slug of [a-z0-9-]+ or the literal "query" if the input collapses to empty.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "query"


def bronze_path(source: str, query: str) -> Path:
    """Compute the bronze JSONL path for one fetch invocation.

    Args:
        source: Source name ("openlibrary" or "google_books").
        query: The free-text query used for the fetch (slugified for the filename).

    Returns:
        `<bronze_dir>/<source>/<YYYY-MM-DD>/<query-slug>.jsonl`.
    """
    settings = get_settings()
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    return settings.bronze_dir / source / today / f"{slugify(query)}.jsonl"


def cmd_fetch(args: argparse.Namespace) -> None:
    """Run one or both source fetchers against a query and append results to bronze."""
    sources = ["openlibrary", "google_books"] if args.source == "both" else [args.source]
    for src in sources:
        out_path = bronze_path(src, args.query)
        if src == "openlibrary":
            docs = openlibrary.fetch_search(args.query, limit=args.limit, out_path=out_path)
        elif src == "google_books":
            docs = google_books.fetch_search(args.query, limit=args.limit, out_path=out_path)
        else:
            raise ValueError(f"unknown source: {src}")
        print(f"[fetch:{src}] {len(docs):>4} records → {out_path}")


def cmd_silver(args: argparse.Namespace) -> None:
    """Parse all bronze JSONLs, dedupe, write silver parquet."""
    out, n = to_silver.run(source=args.source if args.source != "both" else None)
    print(f"[silver] {n:,} unique books → {out}")


def cmd_gold(args: argparse.Namespace) -> None:
    """Embed silver descriptions and build feature vectors."""
    from mybookrec.ingest import to_gold  # local import: torch is heavy

    out_path, n = to_gold.run(model_name=args.model)
    print(f"[gold] {n:,} books with embeddings → {out_path}")


def cmd_refresh(args: argparse.Namespace) -> None:
    """Add new gold items to the live FAISS index."""
    from mybookrec.ingest import refresh_index

    added = refresh_index.run(index_path=args.index)
    print(f"[refresh] +{added:,} items into {args.index}")


def parse_args() -> argparse.Namespace:
    """Build the argparse tree and parse argv.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch raw API responses → bronze JSONL.")
    p_fetch.add_argument("--query", required=True)
    p_fetch.add_argument("--source", choices=["openlibrary", "google_books", "both"], default="both")
    p_fetch.add_argument("--limit", type=int, default=40)
    p_fetch.set_defaults(func=cmd_fetch)

    p_silver = sub.add_parser("silver", help="Bronze → silver: parse, dedupe, write parquet.")
    p_silver.add_argument("--source", choices=["openlibrary", "google_books", "both"], default="both")
    p_silver.set_defaults(func=cmd_silver)

    p_gold = sub.add_parser("gold", help="Silver → gold: embeddings + feature vectors.")
    p_gold.add_argument("--model", default=None, help="HF model id (default: settings.embed_model_name).")
    p_gold.set_defaults(func=cmd_gold)

    p_refresh = sub.add_parser("refresh", help="Append new gold embeddings to a FAISS index.")
    p_refresh.add_argument("--index", type=Path, required=True)
    p_refresh.set_defaults(func=cmd_refresh)

    return parser.parse_args()


def main() -> None:
    """Parse args and dispatch to the selected sub-command."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
