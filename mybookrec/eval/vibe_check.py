"""Generate top-K recommendations for the synthetic 'me' user with diagnostic checks.

Diagnostic checks (against the known synthetic profile in my_books.csv):
    Red flags (should not appear): Twilight, school classics, adult dense fantasy,
        hyped contemporary YA the synthetic user disliked.
    Green flags (should appear): adjacent YA fantasy (Throne of Glass, etc.),
        shoujo manga, middle-grade, Brandon Mull-style children's fantasy.

Usage:
    .venv/bin/python -m mybookrec.eval.vibe_check checkpoints/two_tower_v4bce_best.pt
    .venv/bin/python -m mybookrec.eval.vibe_check <ckpt> --top-k 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch

from mybookrec import DATA_DIR
from mybookrec.io.checkpoints import (
    batch_encode,
    load_checkpoint,
    load_item_features,
    personal_user_features_path,
)

RED_FLAGS: tuple[tuple[str, list[str]], ...] = (
    ("TWILIGHT (1★ × 5 = strongest negative)", ["twilight", "midnight sun", "breaking dawn", "eclipse"]),
    ("School-reading classics (1-2★)", ["wuthering heights", "lord of the flies", "grapes of wrath"]),
    ("Adult dense fantasy (1-2★)", ["elantris", "sword of shannara", "furies of calderon", "shadow of the wind"]),
    (
        "Hyped contemporary YA they bucked (1-2★)",
        ["divergent", "paper towns", "iron king", "grave mercy", "false prince"],
    ),
)
GREEN_FLAGS: tuple[tuple[str, list[str]], ...] = (
    (
        "Adjacent YA fantasy with female leads",
        [
            "throne of glass",
            "court of thorns",
            "daughter of smoke",
            "caraval",
            "shadow and bone",
            "red queen",
            "shatter me",
        ],
    ),
    (
        "Shoujo / coming-of-age manga",
        [
            "fruits basket",
            "skip beat",
            "ouran",
            "kimi ni todoke",
            "lovely complex",
            "honey and clover",
            "vampire knight",
        ],
    ),
    (
        "Middle-grade with young female leads",
        ["where the mountain", "ella enchanted", "tuck everlasting", "girl who", "savvy"],
    ),
    ("Brandon Mull / LDS-adjacent children's fantasy", ["fablehaven", "beyonders", "five kingdoms", "candy shop war"]),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace with `checkpoint` and `top_k`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def find_in_top(
    title_pool: list[tuple[int, str, str]],
    needles: list[str],
) -> list[tuple[int, str, str]]:
    """Return ranked entries whose lowercased title contains any of the needles.

    Args:
        title_pool: List of (rank, lowercase_title, book_id) tuples.
        needles: Lowercase substrings to match.

    Returns:
        Matching subset of title_pool, preserving rank order.
    """
    return [(rank, title[:60], bid) for rank, title, bid in title_pool if any(needle in title for needle in needles)]


def print_diagnostic_section(
    title_pool: list[tuple[int, str, str]],
    flags: tuple[tuple[str, list[str]], ...],
    want_some: bool,
) -> tuple[int, int]:
    """Print and count flag hits across a category list.

    Args:
        title_pool: Top-100 (rank, lowercase_title, book_id) tuples.
        flags: List of (label, needles) tuples to scan for.
        want_some: True = green flags (want >0), False = red flags (want 0).

    Returns:
        Tuple of (total_hits, categories_with_at_least_one_hit).
    """
    total_hits = 0
    categories_hit = 0
    intent = "want some" if want_some else "want NONE"
    for label, needles in flags:
        hits = find_in_top(title_pool, needles)
        total_hits += len(hits)
        categories_hit += 1 if hits else 0
        success = (len(hits) > 0) if want_some else (len(hits) == 0)
        status = "✓" if success else "✗"
        print(f"\n{status} [{intent}] {label}: {len(hits)} found in top-100")
        for rank, title, _ in hits[:5]:
            print(f"    rank {rank}: {title}")
    return total_hits, categories_hit


def main() -> None:
    """Generate top-K recommendations and run diagnostic checks."""
    args = parse_args()

    model, config, _ = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device.type
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    personal_path = personal_user_features_path(config)
    personal_features = torch.from_numpy(np.load(personal_path)).to(device).float().unsqueeze(0)
    item_features = load_item_features(config, device)
    print(f"Personal features: {tuple(personal_features.shape)}  Items: {tuple(item_features.shape)}")

    with torch.no_grad():
        user_emb = model.encode_user(personal_features)
        all_item_embs = batch_encode(model.encode_item, item_features)
        scores = (user_emb @ all_item_embs.T).squeeze(0)

    # Mask already-rated books — recommendations should be new to the user.
    transformed = DATA_DIR / "transformed"
    with open(transformed / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)
    my_books = pl.read_csv(transformed / "my_books.csv")
    rated_idxs = [book_id_to_index[str(bid)] for bid in my_books["book_id"].to_list() if str(bid) in book_id_to_index]
    scores[rated_idxs] = float("-inf")
    print(f"Masked {len(rated_idxs)} already-rated books")

    # Always pull top-100 for diagnostics, display only top_k.
    top_scores, top_idxs = torch.topk(scores, k=max(args.top_k, 100))
    top_idxs_list = top_idxs.cpu().tolist()
    top_scores_list = top_scores.cpu().tolist()
    index_to_book_id = {v: k for k, v in book_id_to_index.items()}
    top_book_ids = [index_to_book_id[i] for i in top_idxs_list]

    meta = (
        pl.scan_parquet(transformed / "books_with_genres.parquet")
        .select("book_id", "title", "num_pages", "average_rating", "genres")
        .filter(pl.col("book_id").is_in(top_book_ids))
        .collect()
    )
    meta_dict = {row["book_id"]: row for row in meta.to_dicts()}

    print(f"\n{'=' * 100}")
    print(f"TOP {args.top_k} RECOMMENDATIONS")
    print(f"{'=' * 100}")
    print(f"{'rank':<5} {'score':<8} {'avg':<6} {'pages':<6} {'title'}")
    print("-" * 100)
    for rank, (bid, score) in enumerate(zip(top_book_ids[: args.top_k], top_scores_list[: args.top_k]), 1):
        row = meta_dict.get(bid, {})
        title = (row.get("title") or "?")[:75]
        avg = row.get("average_rating", "?")
        pages = row.get("num_pages") or "?"
        print(f"{rank:<5} {score:<8.4f} {str(avg)[:5]:<6} {str(pages)[:5]:<6} {title}")

    title_pool = [
        (rank, (meta_dict.get(bid, {}).get("title") or "").lower(), bid)
        for rank, bid in enumerate(top_book_ids[:100], 1)
    ]

    print(f"\n{'=' * 100}\nDIAGNOSTIC CHECKS\n{'=' * 100}")
    red_total, _ = print_diagnostic_section(title_pool, RED_FLAGS, want_some=False)
    green_total, green_categories_hit = print_diagnostic_section(title_pool, GREEN_FLAGS, want_some=True)

    print(f"\n{'=' * 100}\nGENRE DIVERSITY (top-20)\n{'=' * 100}")
    genre_counter: Counter[str] = Counter()
    for bid in top_book_ids[:20]:
        g = meta_dict.get(bid, {}).get("genres") or {}
        if isinstance(g, dict) and g:
            top_genre = max(g.items(), key=lambda kv: kv[1] or 0)[0]
            genre_counter[top_genre] += 1
    for genre, count in sorted(genre_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}× {genre}")

    twilight_count = len(find_in_top(title_pool, RED_FLAGS[0][1]))
    print(f"\n{'=' * 100}\nVERDICT\n{'=' * 100}")
    print(f"Red flags total:   {red_total} (want 0)")
    print(f"Green flags total: {green_total} (want > 0)")
    if twilight_count > 0:
        print("❌ Dislike embedding is broken — Twilight in top-100.")
    elif red_total > green_total:
        print("⚠️  More red than green — taste signal weak or generic.")
    elif green_total == 0:
        print("⚠️  No green hits — vocabulary may not match diagnostic needles. Inspect top-K manually.")
    else:
        print(
            f"✓ Twilight masked, {green_total} green-flag matches across "
            f"{green_categories_hit}/{len(GREEN_FLAGS)} categories."
        )


if __name__ == "__main__":
    main()
