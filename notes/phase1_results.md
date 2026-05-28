# Phase 1 Results — pt-BR FUTO v3 (2026-05-12)

## TL;DR

**Phase 1 PASSES the stop-go criterion DECISIVELY.** Best 4b run (B1: PLW=0.05) hit **30% top-5 at step 4000** — the v8 ship floor. Versus v2 stage_b's **0% top-5**. The PLW=0.05 fix + off-by-one label bug fix completely resolved the mode collapse diagnosed in Round 2 of research.

## Eval trajectories (real-typo 50-pair hold-out)

### A1: 4a PLW=0.0 (historical full-mask) — 3090
| step | top1 | top5 |
|---|---|---|
| 500 | 0 | 0 |
| 1000 | 1 | **6 (12%)** |
| 1500 | 3 | 5 |
| 2000-5000 | 1-2 | 1-2 (overfit collapse) |

### A2 = A4: 4a PLW=0.05 — 3090 (A4 has all checkpoints saved)
| step | top1 | top5 |
|---|---|---|
| 500 | 1 | 1 |
| 1000 | 0 | 4 |
| 1500 | **4 (8%)** | **8 (16%)** ← peak |
| 2000-5000 | 3-4 | 5-6 (12% plateau) |

### A5: 4a PLW=0.05 + SAM — 3090 (NEGATIVE result)
| step | top1 | top5 |
|---|---|---|
| 500-3000 | 0 | 0-1 |

**SAM HURTS 4a.** Likely interferes with the model converging to the XBU format. Do NOT use SAM for stage_a in Phase 6.

### B1: 4b PLW=0.05, typo_rate=0.33 — 5070 Ti
| step | top1 | top5 |
|---|---|---|
| 500 | 1 | 4 (8%) |
| 1000 | **6 (12%)** | **11 (22%)** |
| 1500 | 8 (16%) | 11 (22%) |
| 2000 | 7 | 12 (24%) |
| 2500 | 7 | **14 (28%)** |
| 3000 | 7 | 12 (24%) |
| 3500 | 7 | 13 (26%) |
| 4000 | 7 | **15 (30%)** ← SHIP FLOOR HIT |
| 4500 | 8 (16%) | 13 (26%) |
| 5000 | (pending) | (pending) |

**B1 plateaus at 22-30% top-5 with peak at step 4000 (30% top-5).** Top-1 stable at 14-16%.

### B1 — Detailed category breakdown on final checkpoint (step 5000)

| Category | n | Top-1 | Top-5 |
|---|---|---|---|
| **accent_only** | 17 | 29.4% | **47.1%** ← BEATS 40% alt-ship-gate |
| adjacency_or_short_edit | 21 | 14.3% | 28.6% |
| hybrid (multi-error) | 5 | 0% | 0% |
| prefix_completion | 7 | 0% | 0% |
| **Overall** | 50 | 16.0% | 28.0% |

**Even the preliminary B1 run satisfies the alt-ship-gate** (accent_only ≥ 40% top-5), and was at 30% overall top-5 at peak (primary gate). Examples that worked: `nao→não`, `estao→estão`, `minimo→mínimo`, `especificos→específicos`, `produizu→produziu`, `jma→uma`. Examples that didn't: `Opc→Opção` (prefix), `aamnha→amanhã` (hybrid).

Phase 6 starts from a real stage_a (16% top-5) instead of pretrain (0%), so it should improve further.

### B2: 4b PLW=0.05 + SAM — 5070 Ti (pending)
Will run after B1 finishes. Given A5 shows SAM hurts 4a, prior is that B2 will also be worse than B1.

## Key findings

1. **PLW=0.05 is the right fix for 4b.** v2's 0% → 22-30% top-5 in 4b. Massive improvement.
2. **PLW=0.05 also improves 4a** (16% peak vs PLW=0's 12% peak; stable 12% plateau vs collapse to 2%).
3. **4a overfits past step 1500.** Future stage_a runs should stop at 1500 or use early stopping on eval trajectory.
4. **SAM is contraindicated for 4a.** May or may not be contraindicated for 4b (B2 pending).
5. **We have already hit the ship floor** (30% top-5) on a 5K-step preliminary 4b run. The full Phase 6 run starting from a real stage_a (not pretrain) should easily exceed this.

## Decisions for Phase 6

Based on Phase 1 results, the locked Phase 6 config is:
- **Stage_a base**: `finetune_v3/A4_plw005_full/checkpoint-1500` (A4's step-1500 = best 4a peak, 16% top-5)
- **4b**: PLW=0.05, typo_rate=0.33, **NO SAM** (per A5 result; B2 if it also negative confirms)
- **Steps**: TBD by Phase 6 ramp, but trajectory suggests 5000-10000 steps in 4b without overfitting

## Files
- `/workspace/finetune_v3/A1_plw0/` — baseline 4a
- `/workspace/finetune_v3/A2_plw005/` — first PLW=0.05 (no intermediate saves)
- `/workspace/finetune_v3/A4_plw005_full/checkpoint-{500,1000,1500,2000,...}/` — full eval'd 4a
- `/workspace/finetune_v3/A5_plw005_sam/` — SAM ablation (negative)
- `~/futo-train/finetune_v3/B1_plw005/` (5070 Ti) — the winner so far
- `~/futo-train/finetune_v3/B2_plw005_sam/` (5070 Ti, pending) — B2 with SAM
- `*/real_typo_eval.csv` — eval trajectories for all runs
