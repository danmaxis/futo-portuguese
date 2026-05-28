# Phase 1 Stage A results (3090, COMPLETE 2026-05-12)

## Setup
- Base: `pretrain_big/base` (36M Llama, vocab 15008)
- Dataset: `notes/synth_typos.json` (200K hand-rolled) + `notes/real_typos_pool.json` (343 real), 25% real-mix
- Seq_len: 64, micro_batch: 32, grad_accum: 1 → effective batch 32
- LR: 1e-4 cosine, warmup 200, 5000 steps
- Eval: 50-pair real-typo hold-out, every 500 steps

## Eval trajectories

### A1: PLW=0.0 (historical full-mask = old 04a behavior)
| step | top1 | top5 |
|---|---|---|
| 500 | 0 | 0 |
| 1000 | 1 | **6** |
| 1500 | **3** | 5 |
| 2000 | 2 | 2 |
| 2500 | 2 | 2 |
| 3000 | 2 | 2 |
| 3500 | 1 | 2 |
| 4000 | 1 | 1 |
| 4500 | 1 | 1 |
| 5000 | 1 | 1 |

**Peak: step 1000-1500 (~12% top-5, ~6% top-1). Strong overfitting after.**

### A2: PLW=0.05 (relaxed mask per arxiv 2401.13586)
| step | top1 | top5 |
|---|---|---|
| 500 | 1 | 1 |
| 1000 | 0 | 4 |
| 1500 | **4** | **8** |
| 2000 | 4 | 6 |
| 2500 | 4 | 6 |
| 3000 | 4 | 5 |
| 3500 | 3 | 5 |
| 4000 | 4 | 6 |
| 4500 | 3 | 6 |
| 5000 | 4 | 6 |

**Peak: step 1500 (16% top-5, 8% top-1). Stable plateau at ~12% top-5.**

## Findings

1. **PLW=0.05 beats PLW=0** at every measure: higher peak (16% vs 12% top-5), higher endpoint (12% vs 2%), no overfitting collapse. The prompt-loss-weight literature is directly validated for our short-completion regime.

2. **A1 overfits past step 1500.** v2's "8K-step 4a" (final_loss 0.36) and v2-v3's reported 12% top-5 were both at the OVERFITTING side of the curve. The recipe for stage_a should be ~1500 steps, not 5K-8K.

3. **A2 is stable.** PLW=0.05's regularization keeps top-5 at ~12% across steps 2K-5K. This is a much better property: less sensitive to choosing an exact stop point.

## Recommendations for Phase 6 (final big run)

- **Use PLW=0.05 for stage_a.** Step count: 1500-2000 (not 8K).
- Save every 500 steps so we can pick the best checkpoint by real-typo top-5 trajectory.
- Stage_a's best checkpoint feeds into stage_b (4b), which is still being validated by Phase 1 B1/B2 on the 5070 Ti.

## Files
- `finetune_v3/A1_plw0/` (3090) — final checkpoint at step 5000 (overfit)
- `finetune_v3/A2_plw005/` (3090) — final checkpoint at step 5000 (stable)
- `finetune_v3/A1_plw0/real_typo_eval.csv` and `A2_plw005/real_typo_eval.csv` — full trajectories
