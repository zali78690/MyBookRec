# Deferred improvements — not implemented in v3 overhaul, with concrete next steps

The v3 overhaul implemented 6 of the 10 originally-listed improvements. These 4 were
deferred due to infrastructure constraints, not because they're unimportant. Each has
concrete next steps and an estimated effort.

## Better item embeddings (MPNet 768-dim or BGE-large)

**Why deferred**: requires recomputing description embeddings for all 1.78M books, which
needs a GPU. Local MPS works but takes ~6-10 hours; Colab kernel issues blocked it earlier
in the session.

**Expected lift**: 1.2-1.3× HR@10. Modest but cumulative with other lifts.

**Concrete next steps**:
1. Pick a model: `BAAI/bge-large-en-v1.5` (1024-dim, top of MTEB) or `sentence-transformers/all-mpnet-base-v2` (768-dim, well-trusted).
2. Modify `scripts/embed.ipynb`: replace `EMBEDDING_MODEL = "ibm-granite/granite-embedding-30m-english"` with the new model.
3. Run on Colab T4 (browser UI, not VS Code) or rent an RTX 4090 on vast.ai for ~$0.30/hr (4 hours).
4. Upload to Drive at `data/transformed/book_embeddings_v2.npy`.
5. Update `train_user_features` pipeline to use new embeddings.
6. Retrain. The model architecture only needs `item_input_dim` adjusted (384 → 768 or 1024).

**Risk**: larger embeddings increase memory ~2-3×. On Colab T4 it'll fit. On MPS Mac would
need feature loading rework.

## Review text features

**Why deferred**: UCSD `goodreads_reviews_dedup.json.gz` is ~12 GB compressed, ~50 GB
uncompressed with ~15M reviews. Embedding all of them is a multi-day GPU job. Beyond MVP scope.

**Expected lift**: 1.2-1.5× HR@10. Reviews carry strong taste signal beyond title+description.

**Concrete next steps**:
1. Download `goodreads_reviews_dedup.json.gz` from UCSD (email Mengting Wan).
2. Filter to interactions in our train split only — drops to ~5-7M reviews.
3. For each book, concatenate top-K most-helpful reviews (e.g., 5 reviews, truncate to 512 tokens each).
4. Embed via the same sentence-transformer used in step 11.
5. Add as a fourth item feature channel: `[book_emb | genre | pages | author_emb | review_emb]` → 1163-dim items.
6. Retrain.

**Risk**: review embedding step is ~30 GPU-hours on a T4. ~$15-20 on vast.ai.

## Finer genre taxonomy (BISAC)

**Why deferred**: requires (a) the BISAC taxonomy file from bisg.org (license unclear without
business membership), or (b) building a hybrid taxonomy from UCSD `popular_shelves` field
which is a substantial data-engineering project.

**Expected lift**: 1.2-1.5× HR@10.

**Concrete next steps** (hybrid approach without BISAC license):
1. Process raw books, extract `popular_shelves` field — list of `{count, name}` shelf names per book.
2. Filter shelf names: drop "to-read", "currently-reading", "owned", "favourites" etc. (utility shelves).
3. Embed the remaining shelf names via sentence-transformer (one embedding per unique shelf name, ~50K total).
4. Embed a chosen set of ~50 fine-grained genre labels (e.g. "epic fantasy", "urban fantasy",
   "literary fiction", "historical mystery", ...).
5. For each book, build a soft genre vector: for each genre label, sum `shelf_count × cos_sim(shelf_emb, genre_label_emb)` over all the book's shelves.
6. L2-normalize → 50-dim soft genre vector. Replace existing 10-dim genre vector.
7. Retrain with `item_input_dim` adjusted and `train_user_features` pipeline regenerated.

**Risk**: the shelf-name semantic mapping is fuzzy. Will need EDA to tune the genre label set.

## SASRec (sequence model)

**Why deferred**: architectural rewrite. The user feature would change from a static centroid
to a sequence of (book, rating, date) tuples. The UserTower becomes a transformer encoder
over the user's interaction sequence. Cross-architectural complexity is high.

**Expected lift**: 1.5-2× HR@10. Significant. Captures "user recently into X, drifting away
from Y" — temporal taste shifts that the current centroid approach loses.

**Concrete next steps**:
1. Rebuild `train_user_features` as `(n_users, MAX_SEQ_LEN, per_book_feature_dim)` — a sequence
   of liked-book embeddings rather than a centroid. `MAX_SEQ_LEN = 50` or so.
2. Replace UserTower MLP with a small transformer encoder (e.g., 2 layers, 8 heads, 128-dim).
3. Add positional embeddings (or sinusoidal date encodings).
4. Loss and ItemTower unchanged.
5. Training compute roughly 3-5× the MLP version (transformer attention is quadratic in seq
   length).

**Risk**: largest risk is over-engineering for an MVP. Worth doing only if HR@10 plateaus
on simpler levers AND the use-case demands fresh recommendations matching recent taste shifts.

## Suggested order for these four

If you do one more cycle of improvements after this overhaul, I'd prioritize:

1. **Review text features** — biggest lift (1.2-1.5×) but real cost (GPU + data work).
2. **BISAC / hybrid taxonomy** — 1.2-1.5× and doesn't need GPU.
3. **Better embeddings (BGE/MPNet)** — 1.2-1.3×, ~4 hours of GPU rental.
4. **SASRec** — biggest architectural lift (1.5-2×) but largest engineering investment.

Stacked, these would plausibly add another 2-3× to the post-v3-overhaul HR@10.
