# Evaluation plan — three reference points

To interpret final model quality, we triangulate against three checkpoints:

| Label | Model | Tokenizer | Eval script | When to run |
|---|---|---|---|---|
| **Ceiling** | `reference_model/ml4_1_f16_meta_fixed.gguf` (official English FUTO) | `reference_model/extracted_spm.model` | `scripts/eval_english_baseline.py` | Done — see `notes/eval_english_baseline.json` |
| **Floor** | `pretrain/base/` (pt_BR Wiki-only, NO fine-tuning) | `tokenizer/spm_pt_br.model` | `scripts/eval_keyboard.py` | ✅ Done — `notes/eval_floor_pretrain_base.json` — top-1=0%, top-5=0% (autocorrect), top-1=0%/top-8=0% (next-word) |
| **Final** | `finetune/stage_b/final/` (pt_BR + autocorrect fine-tune) | `tokenizer/spm_pt_br.model` | `scripts/eval_keyboard.py` | ✅ Done — `notes/eval_final_stage_b.json` — top-1=0%, top-5=3.3% (autocorrect). Better than floor on top-5 only. Confirms pipeline works structurally; quality is poor due to small-corpus pretrain. |

## End-to-end validation outcome

**Pipeline works**: `pt_br_futo_v4_final.gguf` loads on FUTO Keyboard pt-BR and produces predictions without crashing.

**Two non-obvious gotchas resolved during side-load**:
1. **GGUF version mismatch**: FUTO's vendored llama.cpp only handles GGUF v2 (28 KV fields). Newer convert_hf_to_gguf.py produces v3 with 9 extra fields → fails to load. Fix: `06b_downgrade_v2.py` strips extras and patches version byte.
2. **Missing `char_embed_mixing_v1` feature**: declaring `xbu_char_autocorrect_v1` alone causes `DecodePromptAndMixes` to read from an empty `LlamaAdapter::embeddings` vector (uninit'd because the embed-mixing feature wasn't requested), producing SIGSEGV at `t.token × n_embd × 4` bytes. Fix: include `char_embed_mixing_v1` in features string. The embedding tensor itself is `token_embd.weight` which is already in any Llama GGUF.

**Quality verdict (mini run on Wiki-pt only)**: poor. As expected — single-domain 500M-token pretrain + 36M params can't ground autocorrect well. To improve:
- Scale corpus to BrWaC + OSCAR + Carolina + Wiki (~3-5B tokens)
- Pretrain to 80-150K steps (vs 20K)
- Possibly: revisit Phase 4a (currently overfits synthetic distribution)

## Floor baseline rationale

The floor model has only seen plain Wikipedia-pt during pretrain — never the
`<XBU><CHAR_*>...<CHAR_*><XBC>...<XEC>` autocorrect format. We expect very poor
autocorrect numbers (top-1 in the 0-15% range) because:

- The model has no training signal for what should follow `<XBC>` after a typo
- The `<CHAR_*>` keypress tokens have effectively random embeddings (all 26 are user-defined symbols included in the SP vocab from day 1, but pretrain text never uses them)
- Next-word prediction in plain pt_BR text MAY work tolerably (the model did learn pt_BR statistics)

This is the floor: "what we get without proper fine-tuning."

## Final model interpretation matrix

After running all three evals, compare top-1 autocorrect:

| Final vs Ceiling | Final vs Floor | Verdict |
|---|---|---|
| >= 90% of ceiling | much better than floor | ✅ **success** — ship it |
| 60-90% of ceiling | much better than floor | ⚠️ acceptable but could improve — extend Phase 4 fine-tune, more epochs, more synthetic typos |
| < 60% of ceiling | much better than floor | 🔧 fine-tune working but undertrained — significantly more Phase 4 steps |
| any | only marginal improvement over floor | 🚨 **fine-tune broken** — debug: check XBU triple format, check label masking in Phase 4a, check loss curve |

## Comparison script (TODO when all three results are in)

`scripts/compare_eval.py` — load three JSON eval outputs, render a table:
- Per-category top-1/top-5 (shortcut, adjacency, transpose, doubled, misspell, trivial)
- Overall pass rates
- Side-by-side
- Color-code: red if final < floor + 10pp, green if final >= ceiling - 10pp

To write after we have data.
