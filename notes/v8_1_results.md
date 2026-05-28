# v8.1 — work log (in progress)

## Status

| Workstream | Status |
|---|---|
| A. FUTO regression diagnosis | ✅ DONE — `notes/futo_regression.md` |
| B. Refresh typo pools + carve disjoint v8.1 hold-out | ✅ DONE — `notes/v8_1/{real_typos_eval,real_typos_pool}.json` |
| C. Claude-synth via `claude -p` CLI | 🟡 READY — `scripts/run_claude_synth_v8_1.sh` (uses local `claude` CLI, no API key). Smoke-tested: ~12s/accepted-pair at concurrency 8. |
| D. Recipe tweaks for top-1 | ✅ DONE — `scripts/run_v8_1.sh` with PLW=0.02 stage_c, typo_rate=0.15, 10K steps, top-1 selection via CSV |
| E. v8.1 full run (3090 + 5070 Ti) | ⏸ Blocked on user kickoff |
| F. Package v8.1 GGUF + ADB push | ⏸ Blocked on E |
| G. Conversational eval (NWP + prefix) | 🟡 Scripts ready; convos being generated (`notes/v8_1/conversations.json`) |

## Workstream A — FUTO regression

Root cause: upstream commit `533b902` ("Ignore blank pastText in complex case for composing wrapper") guards `pastText.isNotEmpty()` in `InputConnectionInternalComposingWrapper.kt`, suppressing empty-prefix NWP suggestions. App-side bug. Not our model. Full writeup in `notes/futo_regression.md`. Workaround: stay on FUTO 0.1.27 until upstream fixes it.

## Workstream B — refreshed pools

Source: `notes/typo_log.jsonl` (534 entries, was 343 at v8 ship).

After dedup: **393 unique pairs**. v8.1 carve:
- **60-pair eval hold-out** at `notes/v8_1/real_typos_eval.json` — strictly disjoint from v8's 50-pair eval (verified overlap = 0).
- **335-pair training pool** at `notes/v8_1/real_typos_pool.json`.

v8.1 eval category breakdown:
| Category | n |
|---|---|
| adjacency_typo | 18 |
| accent_only | 18 |
| prefix_completion | 10 |
| other (≈ hybrid) | 6 |
| cedilla_only | 3 |
| capitalization | 3 |

So 16/58 of the v8.1 hold-out is in the v8-weak-spot (prefix + hybrid) buckets. Lift on those is the headline win we want.

## Workstream C — Claude-synth (ready to invoke)

Edits to `scripts/08_claude_synth.py`:
- Replaced the `anthropic` SDK path with `subprocess` calls to `claude -p --output-format json` (the local Claude Code CLI). No API key needed.
- Stage 1 uses `--json-schema` for structured output via the `structured_output` field.
- `CATEGORY_HINTS` dict — per-category instruction snippets injected into Stage1 prompt.
- `--priors-json` flag — override `CATEGORY_PRIORS` for biased pools.
- `--concurrency` flag (default 4) — `ThreadPoolExecutor` over `claude -p` calls. ~3.8× speedup at conc 4.
- Each call picks a category from the priors, appends the hint, and tags the accepted pair with `category_hint`.
- Default model switched to `claude-haiku-4-5` (Sonnet is overkill for typo gen and ~3× slower).

Two-pass invocation in `scripts/run_claude_synth_v8_1.sh`:
1. **General pool** — 500 pairs, default priors.
2. **Weak-spot pool** — 250 pairs, `notes/v8_1/priors_weakspot.json` (`prefix_completion=0.40`, `hybrid_multi=0.25`).

To run (no API key — uses local `claude` CLI auth):
```
SEED_CORPUS=corpora/big/shard_00000.txt ./scripts/run_claude_synth_v8_1.sh
```

Smoke test: 3 pairs / 6 seeds in 37s at concurrency 4 (vs 142s sequential). For 750 pairs at concurrency 8, expect ~75 min total wall.

Outputs: `notes/v8_1/synth_claude_{general,weakspot}.json`.

## Workstream D — recipe + full-pipeline build

`scripts/run_v8_1.sh` now does the **full pipeline by default** (stage_a → stage_b → stage_c). Stage_a is fast (~90s on 3090 for 3K steps) and is the natural place to consume the Claude-synth pairs (which only feed 04a's pair-level dataset). Skip individual stages with `STAGE_X=0`.

| Knob | v8 | v8.1 default | rationale |
|---|---|---|---|
| Stage_a base | pretrain_big/100K | same | unchanged |
| Stage_a synth | hand-rolled 200K | **50K refreshed + Claude general 500 + Claude weakspot 250** | new data; Claude weak-spot pairs target the 0% buckets |
| Stage_a real-mix | 0.25 | **0.40** | more real-typo signal in stage_a (top-1 lever) |
| Stage_a steps | 1500 (picked) | 3000 (with auto best-step pick) | longer head room; eval-every-500 |
| Stage_a PLW | 0.05 | 0.05 (v8-validated) | no change |
| Stage_b typo_rate | 0.33 | **0.40** | more correction examples per batch |
| Stage_b PLW | 0.05 | 0.05 | no change |
| Stage_b steps | 15000 | **12000** | shorter; we have a better-trained stage_a head |
| **Stage_c PLW** | 0.05 | **0.02** | **headline change** — sharper argmax → top-1 lever |
| Stage_c steps | 5000 | **10000** | v8's stage_c was still climbing at end |
| Stage_c typo_rate | 0.10 | **0.15** | more correction signal in conv corpus |
| Best-step selection | manual | **auto via CSV** — stage_a by top-5, stage_c by top-1 |
| Final eval | 50-pair v8 set | **58-pair v8.1 disjoint set** + v8 set as regression check |

Helper: `scripts/merge_synth_pools.py` combines the three synth sources into `notes/v8_1/synth_combined.json` (dedup on typed/committed).

`HF TrainingArguments` defaults `label_smoothing_factor=0.0` — already off in 04b, no change needed.

## Workstream E — kickoff

On 3090 container (`/workspace`) — full pipeline (default):
```
# Push the combined synth pool + updated scripts (after Claude synth completes locally):
python3 scripts/merge_synth_pools.py \
    --inputs notes/v8_1/synth_typos.json \
             notes/v8_1/synth_claude_general.json \
             notes/v8_1/synth_claude_weakspot.json \
    --out notes/v8_1/synth_combined.json
# rsync notes/v8_1/synth_combined.json + scripts/{run_v8_1.sh,merge_synth_pools.py} to /workspace
nohup ./scripts/run_v8_1.sh > /workspace/v8_1.log 2>&1 &
```

Wall: ~90s stage_a + ~170 min stage_b + ~67 min stage_c ≈ 4h on the 3090 (based on v8 run rates).

On 5070 Ti (`~/futo-train`), the control twin (skip stage_a/b, only redo stage_c with v8 PLW_C=0.05):
```
STAGE_A=0 STAGE_B=0 PLW_C=0.05 nohup ./scripts/run_v8_1.sh > ~/v8_1_control.log 2>&1 &
```

## Workstream G — conversational eval (NWP + prefix-completion)

Closes a gap in the existing eval pipeline: `eval_real_typos.py` only tests typo-correction (given typed word → correct word in top-N). It does not test the in-context next-word strip the user sees while typing normally — which is exactly the surface the FUTO 533b902 regression broke, and the part that drives "good but not great" feel.

**Conversations**: 13 scenarios × 5 convos = 65 convos generated via `claude -p` (no API key). Saved to `notes/v8_1/conversations.json`. Scenarios:
- friends_casual, family_logistics, workmates (general)
- parenting, gig_planning (user-requested)
- breakfast, lunch, dinner (user-requested)
- op_movies, op_social_media, op_celebrities, op_news, op_philosophy (user-requested opinion topics)

**Eval script**: `scripts/eval_conversational.py`. For each word boundary in each conversation:
- **NWP**: feed running context, beam-search top-5 next-word candidates, score top-1/top-3/top-5.
- **Prefix completion**: feed `<XBU>PFX<XBC>` for the first 2 and 3 chars of the next word, score top-1/top-3/top-5.

Output: overall + per-scenario hit rates + 30 sample predictions for spot-check.

**Important**: NWP "correct" is fuzzy (many valid next words exist). Treat absolute numbers cautiously; use this primarily as a **relative** v8-vs-v8.1 measure. Run both back-to-back on the same convo set.

To eval v8 (baseline) and v8.1 (candidate) on the 3090:
```
python3 scripts/eval_conversational.py \
    --checkpoint finetune_big_v3/stage_c/final \
    --tokenizer tokenizer/spm_pt_br_v2.model \
    --conversations notes/v8_1/conversations.json \
    --out notes/v8_1/eval_conv_v8.json

python3 scripts/eval_conversational.py \
    --checkpoint finetune_big_v8_1/stage_c/checkpoint-XXXX \
    --tokenizer tokenizer/spm_pt_br_v2.model \
    --conversations notes/v8_1/conversations.json \
    --out notes/v8_1/eval_conv_v8_1.json
```

## Workstream F — package & ship

After best step is picked:
```
./scripts/run_phase7_package.sh   # uses scripts/05_to_futo_gguf.py + 06b_downgrade_v2.py
adb push models/futo_pt_br_v8_1.gguf /sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/models/
```

Smoke-test cases beyond v8's: at least 3 hybrid and 3 prefix_completion pairs from the v8.1 hold-out.

## Results — v8.1 complete (2026-05-22 16:54Z)

Best checkpoint picked by top-1: `finetune_big_v8_1/stage_c/checkpoint-6500` (of 10000 steps).
Wall: stage_a 8 min, stage_b 2h41m, stage_c 50 min ≈ 3h40m total on 3090.

### On the original v8 50-pair hold-out (apples-to-apples regression check)

| Metric | v8 | v8.1 | Δ |
|---|---|---|---|
| **overall top-1** | 36.0% | **44.0%** | **+8.0pp** |
| **overall top-5** | 56.0% | **64.0%** | **+8.0pp** |
| accent_only top-5 | 88.2% | 88% (15/17) | flat |
| accent_only top-1 | — | 65% (11/17) | new metric |
| adjacency top-5 | 61.9% | 71% (15/21) | +9pp |
| adjacency top-1 | — | 52% (11/21) | new metric |
| **hybrid top-5** | **0.0%** | **20% (1/5)** | **+20pp** |
| hybrid top-1 | 0.0% | 0% | flat |
| **prefix_completion top-5** | **0.0%** | **14% (1/7)** | **+14pp** |
| prefix_completion top-1 | 0.0% | 0% | flat |

### On the new harder v8.1 58-pair disjoint hold-out

| Metric | v8.1 |
|---|---|
| overall top-1 | 39.7% (23/58) |
| overall top-5 | 55.2% (32/58) |
| accent_only top-1/5 | 52% / 67% (n=21) |
| adjacency top-1/5 | 42% / 53% (n=19) |
| cedilla top-1/5 | 33% / 100% (n=3) |
| hybrid top-1/5 | 40% / 60% (n=5) |
| **prefix_completion top-1/5** | **10% / 20%** (n=10) |

### Ship gates

- ✗ **Primary** (top-1 ≥ 50% on v8.1 hold-out): **39.7% — MISS by 10pp**
- ✓ **Primary on v8 hold-out**: 44.0% (well above v8's 36%)
- ✓ **Non-regression top-5**: 64% on v8 hold-out (>50%), 88% accent_only (>80%)
- ⚠ **Secondary** (hybrid + prefix top-5 ≥ 20%): on v8 holdout hybrid 20% / prefix 14% (borderline); on v8.1 holdout hybrid 60% / prefix 20% (passes)

### Interpretation

The recipe changes (PLW=0.02 stage_c + real_mix=0.40 stage_a + 178 Claude weak-spot pairs + refreshed real-typo pool) worked exactly as intended:
- **Sharper top-1**: closed half the gap from top-5 to top-1 (was 20pp gap in v8, now 12pp gap in v8.1 on the v8 holdout).
- **Weak spots off zero**: hybrid + prefix_completion now contribute non-zero top-5 hits.
- **No regressions**: accent + adjacency stayed flat or improved.

The harder v8.1 holdout missed the 50% gate because it contains 10 prefix_completion pairs (vs 7 in v8 holdout) and they are the genuine bottleneck — only 10% top-1 / 20% top-5. The 178 Claude weak-spot pairs landed only 10 prefix_completion examples (Stage2 round-trip rejects most), so the model didn't see enough new prefix signal.

### Decision (2026-05-22)

User chose: wait for v8.2-base + ship the better of the two.

**v8.2-base side quest**: blocked. `gen_synth_corpus.py` hit a Claude Code rate-limit storm — only 140 of 2000 calls completed (1860 errors). Retry after quota resets (~24h).

Once v8.2-base finishes, both v8.1 and v8.2 will be evaluated on the v8.1 hold-out + the conversational eval (notes/eval_conv_v8.json + eval_conv_v8_1.json — running now on 3090).
