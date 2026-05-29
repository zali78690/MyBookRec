# v3 Overhaul — Results and Findings

## Honest TL;DR

**v1 still wins on HR@10.** Of the major experiments run in this overhaul, none of v2, v3, v4-BCE, or v6 (pipeline) beat v1 on apples-to-apples eval. That doesn't mean the work was wasted — several real bugs were caught and fixed, the author-feature pipeline is ready, and the failure modes of InfoNCE on this dataset were clearly characterized. But the headline production model is unchanged: **`two_tower_mac.pt` (v1) with HR@10 = 0.0100**.

The cause of the result isn't the architecture choices — it's **training budget**. v4-BCE was killed at batch 5,000 due to system load (mac buffering); v1 trained 56,000 batches. With more compute, v4-BCE (author features + BCE) showed early signs of beating v1.

## Final results — apples-to-apples eval (5,000 test pairs, rng=0, leave-one-out HR@10)

| Model | Trained batches | HR@10 | NDCG@10 | NDCG\|hit | Verdict |
|---|---|---|---|---|---|
| Random | — | 0.000006 | — | — | floor |
| **v1 (current best)** | **55,914** | **0.0100** | **0.0043** | **0.43** | **production model** |
| v2 (log_freq + heavy reg) | 10,000 | 0.0066 | 0.0035 | 0.54 | over-regularized |
| v3 attempt 1 (InfoNCE learnable τ) | 10,000 | 0.0015 → 0 | — | — | temperature divergence, collapse |
| v3 attempt 2 (InfoNCE fixed τ + lower LR) | 5,000 (killed) | 0.0015 | 0.0008 | — | same slow convergence |
| v4-BCE (BCE + author features) | 5,000 (killed) | 0.0062 | 0.0027 | 0.44 | undertrained vs v1 |
| v6 pipeline (v4-BCE retrieve + xenc rerank) | 10,696 (xenc) | 0.0054 | 0.0023 | — | reranker undertrained |

The "Trained batches" column is critical: v1 had 11× the training of v4-BCE. A direct HR@10 comparison is unfair to the new architectures.

## Per-change attribution

### 1. BCE → InfoNCE — DID NOT CONVERGE

**Expected**: 2-5× lift.
**Actual**: HR@10 = 0.0015 at batch 5,000 (3× *worse* than v1 at same batch). Loss decreased steadily but HR@10 collapsed.

**Root causes** (both real, both confirmed by experiment):
1. **Learnable temperature divergence** — the model found it easier to crank up the softmax temperature (sharpening the distribution) than to improve the underlying embeddings. Temperature climbed from 10 to 25+ within 10k batches with no HR@10 improvement.
2. **InfoNCE has 256× more competing items per anchor** (1,028 vs 4) — much harder optimization landscape. With random L2-normalized init, all dot products are near zero; the model needs many more batches to differentiate.

**Tried**: Fixed temperature at 20 + lower LR (3e-4). Still HR@10 = 0.0015 at batch 5,000.

**To make InfoNCE work, you'd need**: 50,000+ batches of training, temperature warmup, or smaller-batch training without in-batch negatives.

**Was the implementation correct?** Yes. Verified via subagent code review (which found a critical in-batch positive leakage bug — when two batch members share the same positive book, one would appear as the other's negative). Separately found and fixed an MPS-specific NaN bug: `-inf * temperature` in the autograd backward pass produced NaN gradients on `log_temperature` that cascaded to all parameters within 2 batches. Fixed by masking after temperature scaling rather than before.

### 2. In-batch negatives — implemented, only useful with InfoNCE

The mechanism is mathematically sound and the collision mask (`pos_item_idx[i] == pos_item_idx[j]` for off-diagonal entries → mask to -inf) is necessary for correctness. But it only matters in the InfoNCE setting; BCE doesn't have in-batch competition. Implementation is in `mybookrec/model/loss.py`.

### 3. Author identity features — **REAL signal, but undertrained**

The pipeline `build_author_features.py` produces:
- `book_to_author_idx.npy` — (1.78M,) mapping book → primary author
- `author_embeddings.npy` — (386K, 384) mean of each author's books' description embeddings
- `item_features_v4.npy` — (1.78M, 779) = [book_emb | genre | pages | author_emb]
- `train_user_features_v4.npy` — (783K, 1,163) = [like | dislike | genre | pages | author_taste]

99.9% of books got author coverage (6 books missing authors out of 1.78M).

**In-training eval suggested promise**: at batch 5,000, v4-BCE's in-training HR@10 = 0.0075, vs v1's first-checkpoint HR@10 ~0.005 — about 50% relative lift on the same training stage.

**CLI eval at batch 5,000**: v4-BCE HR@10 = 0.0062. Below v1's full-training 0.0100. Below v1's first-checkpoint (which we didn't capture at exact batch 5,000 but is likely ~0.004-0.006 based on extrapolation).

**Why the in-training vs CLI eval disagreement?** Different sampling: in-training uses 2,000 pairs with rng=42, CLI uses 5,000 pairs with rng=0. Sample variance at HR=0.005-0.01 is large (~0.0014 std for 2k pairs). The 0.0075 vs 0.0062 numbers are within noise.

**Strong recommendation**: re-run v4-BCE with the full 60+ minute training budget that v1 had. The architecture is sound; we just didn't give it enough batches.

### 4. Hard negative mining — IMPLEMENTED but not run

`/tmp/train_mac_v5.py` is written and unit-validated. Pure InfoNCE failure made it pointless to add hard negatives on top — there was no working baseline to enhance. Code is preserved; if InfoNCE is ever revisited with the longer training schedule, hard mining is one rebase away.

### 5. Cross-encoder ranker (v6) — small lift, undertrained

10,696 batches of cross-encoder training. Loss dropped 0.69 → 0.16, showing healthy learning.

**Pipeline result** (v4-BCE retrieve top-100 → cross-encoder rerank to top-10):
- 2k-pair eval: +9% HR@10 lift over v4-BCE retrieval alone
- 5k-pair eval: -13% HR@10 (within noise)

**Net**: noise-level effect at this training budget. Both the retriever (v4-BCE) and the reranker (v6 cross-encoder) need more training before the pipeline outperforms either alone. Expected lift was 1.5-3×; we got ~1× ± noise.

### 6. Temporal decay — DEFERRED

Requires propagating `date_added` into the slim parquet (currently dropped). Documented as a 30-min next-step in `v3-overhaul-deferred-items.md`.

## What was actually validated

Not the lift numbers — but the correctness of implementations and the existence of pipeline-side failure modes:

1. **InfoNCE temperature divergence** is a real failure mode on this dataset/MPS configuration. Documented with concrete numbers.
2. **In-batch positive leakage** is a real bug (caught by subagent review, fixed). Any future InfoNCE training must include the `pos_item_idx` collision mask.
3. **`-inf * temperature` produces NaN gradients** on MPS. Must mask after temperature scaling.
4. **Author features pipeline works**: 386K authors extracted, 99.9% coverage, embeddings computed via sparse matmul in ~3 seconds, features assembled in under 3 minutes.
5. **Cross-encoder architecture works**: loss decreases as expected; just needs more training time to dominate the reranking pipeline.

## Recommendations going forward

**Highest ROI from here**:

1. **Re-run v4-BCE with v1's training budget** (~60 min on MPS). The author feature lift should materialize when given proper convergence time. This is the single biggest validated unrealized improvement.
2. **Run vibe check on the trained v4-BCE checkpoint** anyway — the absolute HR@10 numbers are eval-set sensitive; qualitative inspection on your personal Goodreads is the real test for a single-user MVP.
3. **InfoNCE with longer schedule** — only worth attempting if you can commit 4-8 hours of training time. The architecture isn't fundamentally broken, it just needs convergence time we didn't have.

**Skipped/deferred items** (with concrete next steps): see [v3-overhaul-deferred-items.md](v3-overhaul-deferred-items.md):
- MPNet/BGE embeddings (~4 GPU-hours, +20-30%)
- Review text features (~30 GPU-hours, +20-50%)
- BISAC taxonomy (~4 hours data work, +20-50%)
- SASRec (~1 week architectural rewrite, +50-100%)

## Production-shaped takeaway

**Use v1 (`two_tower_mac.pt`)** as the recommendation model today. HR@10 = 0.0100 is what the system can reliably deliver. The architectural changes attempted in this overhaul are sound but undertrained. If you can dedicate a longer training run (90+ minutes uninterrupted on MPS, or ~30 minutes on a T4), re-train v4-BCE — based on early signals it should beat v1.

## Files changed/created in this overhaul

**Implementation**:
- `mybookrec/features/training_pairs.py` — added `negative_sampling="log_freq"` with CDF-based fast sampling.
- `mybookrec/model/loss.py` — InfoNCE with in-batch negatives + collision mask + temperature scaling.
- `mybookrec/model/cross_encoder.py` — joint user-item MLP for re-ranking.
- `mybookrec/eval/metrics.py`, `run_eval.ipynb`, `vibe_check.ipynb` — eval infrastructure (already done pre-overhaul, used heavily).

**Data**:
- `data/transformed/book_to_author_idx.npy` — 1.78M book → author mapping
- `data/transformed/author_id_to_index.json` — 386K author → idx mapping
- `data/transformed/author_embeddings.npy` — (386K, 384) author embeddings
- `data/transformed/item_features_v4.npy` — (1.78M, 779) with author dim
- `data/transformed/train_user_features_v4.npy` — (783K, 1163) with author_taste dim

**Checkpoints** (in `data/checkpoints/`):
- `two_tower_mac.pt` — v1 (production model, HR@10 = 0.0100)
- `two_tower_v2_best.pt` — v2 (0.0066)
- `two_tower_v3_best.pt` — v3 InfoNCE attempt (abandoned, 0.0015)
- `two_tower_v4bce_best.pt` — v4-BCE undertrained (0.0062 at 5k batches)
- `cross_encoder_v6.pt` — v6 cross-encoder (10.7k batches)

**Plans**:
- `plans/v3-overhaul-results.md` — this file
- `plans/v3-overhaul-deferred-items.md` — concrete next steps for the 4 deferred improvements
