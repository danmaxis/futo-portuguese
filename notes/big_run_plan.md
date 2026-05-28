# Big run plan — pt_BR FUTO model v2

After the mini run validated the pipeline (Wikipedia-pt 500M tokens → side-loaded GGUF works on phone but suggestions are nonsense), this plan scales corpus + training to actually produce a usable keyboard model.

## Corpus design

Total target: **3-5B tokens**. Mix tilted toward conversational since this is for a phone keyboard.

| Source | Style | Approx pt_BR tokens | Role |
|---|---|---|---|
| BrWaC | Brazilian web (mixed) | ~2.7B | Bulk language model — broad pt_BR statistics |
| OSCAR-pt | Web crawl (mixed) | 10B+ | Backup bulk; cap at 1-2B |
| Wikipedia-pt | Formal/expository | ~500M | Background factual grounding |
| Carolina | Brazilian academic | ~800M | Formal Brazilian register |
| **OpenSubtitles pt-BR** | **Casual film/TV dialogue** | **~150-300M** | **Primary "how Brazilians talk" signal** |
| CORAA | Brazilian conversational speech transcripts | ~10M | Real spoken pt-BR |
| Common Voice pt | Voice-transcribed sentences | ~50M | Short casual utterances |

**Why subtitles are the strongest casual signal**: they capture how people speak in Portuguese — first-person, contractions (`tô`, `tá`, `vc`), present-tense, short utterances, emotional/informal register. Wikipedia is the opposite (third-person, expository, no one types this on a phone).

**Acquisition**: scripts/01_build_corpus.py now has `--datasets big` which pulls all three groups. Run on `gpu-5070ti` (31 GiB DDR4, no swap pressure).

```bash
ssh gpu-5070ti 'cd /home/danmaxis/futo-pt-br && source env/bin/activate && \
  systemd-run --user --scope --unit=corpus-big bash -c "\
    python scripts/01_build_corpus.py \
      --datasets big \
      --target-tokens 4000000000 \
      --out corpora/big \
      > corpora_big.log 2>&1"'
```

Expect 1-3h to assemble.

## Two-stage pretrain (idea)

Single-pass pretraining on the mix would let the bulk web corpus dominate. Better:

**Stage A — broad language model** (~80-120k steps):
- Train on PRIMARY sources only (Wikipedia + BrWaC + OSCAR + Carolina ≈ 3-4B tokens)
- This establishes solid pt_BR statistics
- Output: `pretrain/base_broad/`

**Stage B — conversational adaptation** (~10-20k steps):
- Continue from `base_broad/`, train on CONVERSATIONAL sources only (~300-500M tokens, 1-3 epochs)
- LR drops to 5e-5 (vs 3e-4 in Stage A), warmup 200 steps
- This biases the model toward typing-style register without erasing the broad statistics
- Output: `pretrain/base_casual/` — this is what feeds Phase 4

## Tokenizer

Re-train SentencePiece on a 1-2 GB SAMPLE of the big corpus (mostly the conversational + Brazilian web subset, since the structural symbols have to encode actual user typing). `gpu-5070ti` did the mini run in 14 min; for the big run with `input_sentence_size=2_000_000`:

```bash
shuf corpora/big/shard_*.txt | head -c 1500000000 > /tmp/spm_sample.txt
python scripts/02_train_tokenizer.py --corpus /tmp/spm_sample.txt --out tokenizer/spm_pt_br_v2
```

Slot map stays identical (300 user-defined symbols at 4..303, CHAR_A-Z at 182..207, etc.).

## Phase 4 fine-tune adjustments

Mini-run failure mode: Phase 4a converged too fast on synthetic typos (final loss 0.40), then Phase 4b mode-collapsed when seeing real text mixed with corrections. Adjustments:

- **Phase 4a**: drop steps from 20K → 8K, raise LR floor (cosine never below 5e-5), reduce typos_per_word from 4 to 2. Goal: seed the format without overfitting.
- **Phase 4b**: keep at 20-30K but lower typo_rate from 0.33 to 0.20 — fewer corrections per sentence preserves the language fluency.
- **Phase 4c (NEW for big run)**: smaller stage on conversational corpus only with typo_rate 0.10. ~5K steps. Pushes the model toward casual-register predictions.

## GGUF packaging — known requirements (learned the hard way in mini run)

These MUST match for the FUTO Android app to load and not crash:

1. **GGUF version 2** (newer convert produces v3, the FUTO app's vendored llama.cpp can't read it). Use `scripts/06b_downgrade_v2.py`.
2. **Strip 9 newer KV fields** that the older parser chokes on (handled by 06b too).
3. **Features string MUST include `char_embed_mixing_v1`** alongside `xbu_char_autocorrect_v1`. Without it, `LlamaAdapter::embeddings` stays empty and `DecodePromptAndMixes` SIGSEGVs on every keystroke.
4. **`output.weight` quantized to Q6_K** to match reference (use `llama-quantize --output-tensor-type q6_k`).
5. **`keyboardlm.ext_tokenizer_data` as UINT8 array**, NOT INT32 (use `add_key_value(..., sub_type=GGUFValueType.UINT8)` — Python `bytes` as the value).

All five are baked into our scripts now.

## Wall-clock budget on the 3090

| Phase | Steps | Time |
|---|---|---|
| 1 corpus | — | 1-3 h |
| 2 tokenizer | — | ~15 min on 5070 Ti host |
| 3a pretrain broad | 100k | ~30-50 h |
| 3b pretrain casual | 15k | ~5-8 h |
| 4a fine-tune isolated | 8k | ~20 min |
| 4b fine-tune fulltext | 25k | ~5 h |
| 4c fine-tune casual | 5k | ~1 h |
| 5 GGUF + side-load | — | ~10 min |
| **Total** | | **~2.5-3.5 days** |

## Eval expectation

Reference English: 74% top-1 / 89% top-5. With the bigger pretrain corpus + better-tuned Phase 4, we should target 50%+ top-1 / 75%+ top-5 on the equivalent pt_BR tests. Anything < 30% top-1 means we still have a fundamental issue (corpus quality, fine-tune design, or model size).
