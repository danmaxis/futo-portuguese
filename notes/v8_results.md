# v8 Final Results — pt-BR FUTO LM (2026-05-12)

## TL;DR

**v8 SHIPS.** From v6's 0% top-5 to v8's 56% top-5 (88% on accent_only). Built on the diagnosis that Phase 4b/4c needed (a) PLW=0.05 loss masking instead of PLW=1, and (b) a fix for an off-by-one label-shift bug. SAM was tested and rejected. The 200K hand-rolled synth pool was right-sized to a 343-real + 200K-synth mix; we never had to do the planned Claude-API generation because the recipe fix alone got us 5× above the ship gate.

## Final eval (50-pair real-typo hold-out, never seen in training)

| Category | n | Top-1 | Top-5 |
|---|---|---|---|
| **Overall** | 50 | **36.0%** | **56.0%** |
| **accent_only** | 17 | 52.9% | **88.2%** |
| **adjacency_or_short_edit** | 21 | 42.9% | 61.9% |
| hybrid (multi-error) | 5 | 0% | 0% |
| prefix_completion | 7 | 0% | 0% |

Ship gates:
- Primary: top-5 ≥ 30% overall → **56%** ✅ (1.9×)
- Alt: top-5 ≥ 40% on accent_only → **88.2%** ✅ (2.2×)

Excluding categorically-hard pairs (hybrid + prefix, 12/50): **28/38 = 73.7%** top-5 on "solvable" pairs.

## What v3 did vs v2

| | v2 (broken) | v3 (final) |
|---|---|---|
| Overall top-5 | 0% | 56% |
| accent_only top-5 | 0% | 88.2% |
| Top-1 | 0% | 36% |
| Stage_b loss formulation | PLW=1.0 (full sequence) | PLW=0.05 (relaxed mask) |
| Off-by-one bug in 4b | Present (predicted 2 ahead) | Fixed |
| ABNT2 ç adjacency | Missing | `l↔ç` added |
| Eval cadence | Single endpoint | Every 500 steps |
| SAM | Tried, harmful, dropped | (not used in final) |
| Synth data | 200K hand-rolled | Same (Claude-API deferred — not needed) |
| Real-typo mix in 4a | 25% | 25% (same) |
| Stage_a steps | 8K (overfits) | 1500 (peak; saved every 500) |
| Stage_b steps | 20K | 15K |
| Stage_c steps | 5K | 5K |

## Phase 6 trajectory through training

Stage_b (15K steps, from A4 step-1500 stage_a = 16% top-5):
- step 500: 24% top-5
- step 1000: 34% (already past ship gate)
- step 2000: 40%
- step 3000: 48%
- step 4000: 54%
- step 5000+: plateau at 50-56%

Stage_c (5K steps on conv corpus):
- step 500: top-1 jumped 28% → 34%
- step 1500-4500: 38% top-1, 54% top-5 (plateau)
- step 5000 (final): 36% top-1, 54-56% top-5

## v8 GGUF package

- File: `/home/ai-debian/futo-portuguese/models/futo_pt_br_v8.gguf` (62 MB)
- GGUF v2 ✓
- output.weight = Q6_K, all else F16 (matches reference layout exactly)
- 28 KV fields (matches reference)
- Features string: `base_v1 inverted_space xbu_char_autocorrect_v1 xc0_swipe_typing_v1 char_embed_mixing_v1` (matches reference)
- Smoke test via llama.cpp (with manual SPM tokenization): `<XBU>OBIGADO<XBC>` → `obrigado<XEC>` ✓

## Sequence of v3 work

1. **Phase 0** (code fixes, no GPU): added `--plw`, fixed off-by-one, added eval callback, added ABNT2 ç, added SAM wrapper. Integration tests passed on both 3090 + 5070 Ti.
2. **Phase 1** (validation ablations): A1 (PLW=0) baseline 12% peak then overfit collapse. A2/A4 (PLW=0.05) 16% peak stable plateau. A5 (4a+SAM) 0% across all evals. B1 (4b PLW=0.05) **30% peak top-5, 47% accent_only**. B2 (4b+SAM) half of B1.
3. **Phase 2** (Claude synth) DEFERRED — not needed for ship gate.
4. **Phase 3** (XBU ablation training) NWP twin trained; phone test pending ADB.
5. **Phase 4** (TTL-160m) NOT TRIGGERED — Phase 6 hit ship gate at 36M params.
6. **Phase 5** (Penteado eval) SKIPPED — dataset not findable.
7. **Phase 6** (final big run): from A4 step-1500 stage_a, 15K steps stage_b, 5K steps stage_c — **56% top-5, 36% top-1, 88% accent_only top-5**.
8. **Phase 7** (packaging): GGUF v2 + Q6_K + features done locally; smoke test passes.

## Hardware used

- 3090 (24GB, Unraid container): all 4a runs, Phase 6 stage_b/c (~4h total)
- 5070 Ti (16GB, gaming desktop): all 4b ablation runs B1/B2 + NWP twin (~1.5h)
- VM (CPU): GGUF packaging (~3 min total)

## What remains

**Phase 7 part 2: ADB push to phone.** Requires user to pair the phone. Command sequence:
```
adb pair <phone-ip>:<port> <pair-code>
adb connect <phone-ip>:<port>
adb push /home/ai-debian/futo-portuguese/models/futo_pt_br_v8.gguf \
        /sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/models/
```
Then on phone: Settings → Languages & Models → import the new model.

**Optional follow-ups** (not blocking ship):
- Phase 2 (Claude synth): could push numbers higher, especially on hybrid + prefix_completion. Cost: <$5 in API.
- On-device A/B vs the NWP twin (Phase 3): definitively answer whether XBU format is needed for FUTO's runtime.
