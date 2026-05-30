"""Generate top-K recommendations for the synthetic 'me' user and apply diagnostic checks.

Diagnostic checks (against the known synthetic profile in my_books.csv):
- Red flags: Twilight, school classics, adult dense fantasy, hyped contemporary YA.
- Green flags: adjacent YA fantasy (Throne of Glass etc.), shoujo manga, middle-grade, Brandon Mull.

Usage:
    uv run python scripts/vibe_check.py checkpoints/two_tower_v4bce_best.pt
    uv run python scripts/vibe_check.py <path> --top-k 30
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
from mybookrec.model.towers import TwoTowerModel


# Diagnostic needles: each tuple is (label, list_of_lowercase_substrings, expected_count_pattern).
# expected=False means "want 0 in top-100"; True means "want at least one".
RED_FLAGS = [
    ("TWILIGHT (1★ × 5 = strongest negative)", ["twilight", "midnight sun", "breaking dawn", "eclipse"]),
    ("School-reading classics (1-2★)", ["wuthering heights", "lord of the flies", "grapes of wrath"]),
    ("Adult dense fantasy (1-2★)", ["elantris", "sword of shannara", "furies of calderon", "shadow of the wind"]),
    ("Hyped contemporary YA they bucked (1-2★)", ["divergent", "paper towns", "iron king", "grave mercy", "false prince"]),
]
GREEN_FLAGS = [
    ("Adjacent YA fantasy with female leads", ["throne of glass", "court of thorns", "daughter of smoke", "caraval", "shadow and bone", "red queen", "shatter me"]),
    ("Shoujo / coming-of-age manga", ["fruits basket", "skip beat", "ouran", "kimi ni todoke", "lovely complex", "honey and clover", "vampire knight"]),
    ("Middle-grade with young female leads", ["where the mountain", "ella enchanted", "tuck everlasting", "girl who", "savvy"]),
    ("Brandon Mull / LDS-adjacent children's fantasy", ["fablehaven", "beyonders", "five kingdoms", "candy shop war"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_features_for_checkpoint(config: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    t = DATA_DIR / "transformed"
    if config["user_input_dim"] == 1163:
        personal_path = t / "user_features_v4.npy"
        item_features_np = np.load(t / "item_features_v4.npy")
    else:
        personal_path = t / "user_features.npy"
        emb = np.load(t / "book_embeddings.npy").astype(np.float32)
        genre = np.load(t / "genre_matrix.npy").astype(np.float32)
        pages = np.load(t / "num_pages_normalized.npy").astype(np.float32).reshape(-1, 1)
        item_features_np = np.concatenate([emb, genre, pages], axis=1)

    personal = torch.from_numpy(np.load(personal_path)).to(device).float().unsqueeze(0)
    items = torch.from_numpy(item_features_np.astype(np.float32)).to(device).float()
    return personal, items


def find_in_top(title_pool: list[tuple[int, str, str]], needles: list[str]) -> list[tuple[int, str, str]]:
    """Return (rank, title, book_id) entries whose title contains any of the needles."""
    hits = []
    for rank, title, bid in title_pool:
        if any(needle in title for needle in needles):
            hits.append((rank, title[:60], bid))
    return hits


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}  Checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = TwoTowerModel(
        user_input_dim=config["user_input_dim"],
        item_input_dim=config["item_input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    personal_features, item_features = load_features_for_checkpoint(config, device)
    print(f"Personal features: {tuple(personal_features.shape)}  Items: {tuple(item_features.shape)}")

    with torch.no_grad():
        user_emb = model.encode_user(personal_features)
        item_embs = []
        for i in range(0, item_features.shape[0], 8192):
            item_embs.append(model.encode_item(item_features[i:i + 8192]))
        all_item_embs = torch.cat(item_embs, dim=0)
        scores = (user_emb @ all_item_embs.T).squeeze(0)

    # Mask already-rated books
    t = DATA_DIR / "transformed"
    with open(t / "book_id_to_index.json") as f:
        book_id_to_index = json.load(f)
    my_books = pl.read_csv(t / "my_books.csv")
    rated_idxs = [
        book_id_to_index[str(bid)]
        for bid in my_books["book_id"].to_list()
        if str(bid) in book_id_to_index
    ]
    scores[rated_idxs] = float("-inf")
    print(f"Masked {len(rated_idxs)} already-rated books")

    # Pull top-100 always (diagnostics scan top-100), display top_k
    top_scores, top_idxs = torch.topk(scores, k=max(args.top_k, 100))
    top_idxs_list = top_idxs.cpu().tolist()
    top_scores_list = top_scores.cpu().tolist()
    index_to_book_id = {v: k for k, v in book_id_to_index.items()}
    top_book_ids = [index_to_book_id[i] for i in top_idxs_list]

    meta = (
        pl.scan_parquet(t / "books_with_genres.parquet")
        .select("book_id", "title", "num_pages", "average_rating", "genres")
        .filter(pl.col("book_id").is_in(top_book_ids))
        .collect()
    )
    meta_dict = {row["book_id"]: row for row in meta.to_dicts()}

    # Display top-K
    print(f"\n{'=' * 100}")
    print(f"TOP {args.top_k} RECOMMENDATIONS")
    print(f"{'=' * 100}")
    print(f"{'rank':<5} {'score':<8} {'avg':<6} {'pages':<6} {'title'}")
    print("-" * 100)
    for rank, (bid, score) in enumerate(zip(top_book_ids[:args.top_k], top_scores_list[:args.top_k]), 1):
        row = meta_dict.get(bid, {})
        title = (row.get("title") or "?")[:75]
        avg = row.get("average_rating", "?")
        pages = row.get("num_pages") or "?"
        print(f"{rank:<5} {score:<8.4f} {str(avg)[:5]:<6} {str(pages)[:5]:<6} {title}")

    # Diagnostic checks against synthetic profile
    title_pool = [
        (rank, (meta_dict.get(bid, {}).get("title") or "").lower(), bid)
        for rank, bid in enumerate(top_book_ids[:100], 1)
    ]

    print(f"\n{'=' * 100}\nDIAGNOSTIC CHECKS\n{'=' * 100}")
    red_total = 0
    for label, needles in RED_FLAGS:
        hits = find_in_top(title_pool, needles)
        red_total += len(hits)
        status = "✓" if len(hits) == 0 else "✗"
        print(f"\n{status} [want NONE] {label}: {len(hits)} found in top-100")
        for rank, title, _ in hits[:5]:
            print(f"    rank {rank}: {title}")

    green_total = 0
    for label, needles in GREEN_FLAGS:
        hits = find_in_top(title_pool, needles)
        green_total += len(hits)
        status = "✓" if len(hits) > 0 else "✗"
        print(f"\n{status} [want some] {label}: {len(hits)} found in top-100")
        for rank, title, _ in hits[:5]:
            print(f"    rank {rank}: {title}")

    # Diversity diagnostic
    print(f"\n{'=' * 100}\nGENRE DIVERSITY (top-20)\n{'=' * 100}")
    genre_counter: Counter[str] = Counter()
    for bid in top_book_ids[:20]:
        g = meta_dict.get(bid, {}).get("genres") or {}
        if isinstance(g, dict) and g:
            top_genre = max(g.items(), key=lambda kv: kv[1] or 0)[0]
            genre_counter[top_genre] += 1
    for genre, count in sorted(genre_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}× {genre}")

    # Final verdict
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
        print(f"✓ Twilight masked, {green_total} green-flag matches across {sum(1 for l, _ in GREEN_FLAGS if find_in_top(title_pool, _))}/{len(GREEN_FLAGS)} categories.")


if __name__ == "__main__":
    main()
