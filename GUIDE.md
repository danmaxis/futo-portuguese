# Training a FUTO Keyboard transformer for your language — a practical guide

This document is for anyone who wants to train and side-load a custom transformer language model into [FUTO Keyboard](https://keyboard.futo.org) so they can have **private, on-device, autocorrect + next-word prediction** for a language FUTO doesn't ship a model for.

FUTO ships an English-only ~36M-parameter Llama transformer. Their official position is "we're working on other languages", with no firm ETA. They explicitly support importing third-party models in the same format, but the format spec on their public wiki is incomplete and the integration has several non-obvious gotchas. This guide closes those gaps.

It is written from a real end-to-end run: training a Brazilian Portuguese model, hitting every wall, and shipping a side-loaded `.gguf` that loads and runs on a real Android phone. Where the FUTO public wiki is right, we cite it. Where it's wrong or incomplete, we say so and provide the verified answer.

This guide assumes you can:

- read Python and C++ at a level sufficient to follow training scripts and a crash backtrace,
- operate a Linux box with an NVIDIA GPU,
- use SSH, Docker, and Android Debug Bridge (`adb`).

If you're new to ML training itself, this guide alone won't be enough — it focuses on the FUTO-specific parts.

---

## Table of contents

1. [What FUTO's transformer actually is](#1-what-futos-transformer-actually-is)
2. [Hardware and time budget](#2-hardware-and-time-budget)
3. [The undocumented spec: what your model has to look like](#3-the-undocumented-spec-what-your-model-has-to-look-like)
   1. [Architecture (the easy part)](#31-architecture-the-easy-part)
   2. [The tokenizer: 300 user-defined symbols at fixed slots](#32-the-tokenizer-300-user-defined-symbols-at-fixed-slots)
   3. [The keypress prompt format (NOT literal text)](#33-the-keypress-prompt-format-not-literal-text)
   4. [GGUF metadata fields](#34-gguf-metadata-fields)
4. [Pipeline overview](#4-pipeline-overview)
5. [Phase 0 — extract the official model as your spec](#5-phase-0--extract-the-official-model-as-your-spec)
6. [Phase 1 — assemble a target-language corpus](#6-phase-1--assemble-a-target-language-corpus)
7. [Phase 2 — train the SentencePiece tokenizer](#7-phase-2--train-the-sentencepiece-tokenizer)
8. [Phase 3 — pretrain the base model](#8-phase-3--pretrain-the-base-model)
9. [Phase 4 — autocorrect fine-tune (3 stages)](#9-phase-4--autocorrect-fine-tune-3-stages)
10. [Phase 5 — package as a FUTO-compatible GGUF](#10-phase-5--package-as-a-futo-compatible-gguf)
11. [Side-loading and reproducing a crash](#11-side-loading-and-reproducing-a-crash)
12. [The five gotchas, in one place](#12-the-five-gotchas-in-one-place)
13. [Evaluation methodology](#13-evaluation-methodology)
14. [What this guide does not cover](#14-what-this-guide-does-not-cover)
15. [Acknowledgements and licensing](#15-acknowledgements-and-licensing)

---

## 1. What FUTO's transformer actually is

FUTO Keyboard runs two prediction algorithms in parallel and merges them:

1. A classical AOSP-style dictionary-and-bigram engine. Your language probably already has a dictionary file shipped (`dictionaries/<lang>_wordlist.combined.gz` in [futo-org/android-keyboard](https://github.com/futo-org/android-keyboard)).
2. A **Llama-architecture transformer** trained for autocorrect and next-word prediction. The transformer is ~36M parameters, runs on-device via [llama.cpp](https://github.com/ggerganov/llama.cpp), uses an embedded SentencePiece tokenizer, and only ships in English today.

If you only enable (1), you get spellcheck-grade predictions: word completions, simple autocorrect, learned bigram. Adequate for many languages. If you also enable (2) with a properly-trained model in your language, you get context-aware autocorrect and grammatical next-word prediction.

This guide is about producing (2).

The wiki entry pointing at the format is at <https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs>. Read it once. Then come back here, because several of its claims are misleading and one critical part is missing entirely.

---

## 2. Hardware and time budget

**Minimum to ship a working model**: one consumer NVIDIA GPU with **16+ GiB VRAM**. A 24 GiB GPU is more comfortable. CPU training is impractical at this size.

**Realistic wall-clock for a real-quality run** on one RTX 3090:

| Phase | Time |
|---|---|
| Corpus assembly (3-5B tokens) | 1-3 hours, network-bound |
| Tokenizer training | 15-60 minutes, CPU + RAM-bound |
| Pretrain (~100k optimizer steps) | 30-50 hours |
| Fine-tune (Phase 4a + 4b + 4c) | 3-8 hours |
| GGUF assembly + side-load | <30 minutes |

A **mini validation run** (single corpus source, 20k pretrain steps) takes about 14 hours total and is enough to verify your pipeline end-to-end before committing days to a real run.

**RAM**: SentencePiece tokenizer training is memory-bound and not GPU-accelerated. For a multi-GB corpus you want **32+ GiB RAM** on the host doing tokenizer training. A 16 GiB host will swap heavily and slow to a crawl. See [Phase 2](#7-phase-2--train-the-sentencepiece-tokenizer).

**Disk**: 50-100 GiB free for corpus shards + checkpoints. Heavy artifacts can live on a separate machine accessed over SSH; only the final ~62 MB GGUF needs to leave the training rig.

---

## 3. The undocumented spec: what your model has to look like

Before writing any code: your goal is a `.gguf` file that the FUTO Android app validates and loads at runtime. The validation is strict and partly opaque. The fastest path to getting it right is to **extract the reference English model and reproduce its layout**.

```bash
hf download breadlicker45/futo-keyboard-lm --local-dir reference_model/
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
       reference_model/ml4_1_f16_meta_fixed.gguf > reference_metadata.txt
```

`reference_metadata.txt` is the source of truth. Every section below cites it.

### 3.1 Architecture (the easy part)

A vanilla Llama config:

```python
LlamaConfig(
    vocab_size=15008,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,        # MHA, no GQA
    max_position_embeddings=2048, # NOT 512 — wiki implies 512 but reference is 2048
    rms_norm_eps=1e-6,            # NOT 1e-5 — wiki was wrong
    rope_theta=10000.0,
    tie_word_embeddings=False,
)
```

This is exactly the reference. ~36M parameters total. RoPE is standard llama.cpp defaults. **Do not change these** — the C++ at `LlamaAdapter` validates a few of them.

### 3.2 The tokenizer: 300 user-defined symbols at fixed slots

This is the part the wiki gets most wrong. The tokenizer has **vocabulary size 15008**, broken down as:

| ID range | Count | Contents |
|---|---|---|
| 0..3 | 4 | `<pad>`, `<s>`, `</s>`, `<unk>` (control + unknown) |
| 4..303 | **300** | **User-defined symbols (the layout matters — see below)** |
| 304..559 | 256 | Byte-fallback `<0x00>`..`<0xFF>` |
| 560..15007 | 14448 | BPE pieces learned from corpus (your language fills this) |

The 300 user-defined symbols are a structured layout, not just any 300 strings. Reading the reference and the FUTO Android C++ source at `native/jni/src/ggml/LanguageModel.cpp` and `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` reveals these constraints:

```
Indices 4..27   — <FUTO0>..<FUTO23>  (24 reserved/inert filler slots)
Indices 28..173 — content slots (English contractions/words in reference)
                  → REPLACE with your-language equivalents
Indices 174     — <XBU>     STRUCTURAL: autocorrect "begin user input"
Indices 175     — <XBC>     STRUCTURAL: autocorrect "begin correction"
Indices 176     — <XEC>     STRUCTURAL: autocorrect "end correction"
Indices 177..181— <XC0>..<XC4>  (only XC0 is referenced; rest are reserved)
Indices 182..207— <CHAR_A>..<CHAR_Z>  STRUCTURAL: per-keypress tokens
                  ⚠️ THESE 26 IDs MUST BE CONTIGUOUS AND SEQUENTIAL.
                  The C++ at LanguageModel.cpp resolves <CHAR_A> by name
                  but computes the rest as LETTERS_TO_IDS[0] + i.
Indices 208..263— more content slots (REPLACE with your-language)
Indices 264..303— emoji set (40 slots; you can keep or curate)
```

#### Resolution behavior in the C++ (verified, not just guessed)

The Android library uses two different lookup mechanisms:

- **By name** (via `spm.PieceToId(...)` on the embedded SentencePiece): `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, `<CHAR_A>`, `▁` (SP space marker). Their ID can be anywhere as long as the name resolves to a non-zero index.
- **By computed index** (pointer arithmetic): `<CHAR_B>` through `<CHAR_Z>` are read as `LETTERS_TO_IDS[0] + i`. So the 26 `<CHAR_*>` symbols **must occupy 26 contiguous IDs** in the SentencePiece. The simplest way to guarantee this is to list them sequentially in `user_defined_symbols` — SentencePiece preserves declaration order.

If any of `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, `<CHAR_A>` resolves to 0 (= `<unk>`), the C++ asserts and crashes at model load. So they all must be present in your tokenizer.

#### The `__FUTO0..23` filler slots and emoji

These are inert — no code references them by name and no embedding lookup uses their indices specifically. You can keep them as parity with the reference (`<FUTO0>`, `<FUTO1>`, ...), or replace them with high-frequency words from your language. The reference has English contractions in the content slots; you'd put your language's frequent words/contractions there.

### 3.3 The keypress prompt format (NOT literal text)

This is the **single most important undocumented detail**. The wiki snippet shows it but is easy to misread:

> `This is some <XBU><CHAR_T><CHAR_X><CHAR_E><CHAR_T><XBC>text <XEC>`

The model is **not** prompted with `<XBU>typo<XBC>`. It is prompted with the typed-key sequence as discrete `<CHAR_X>` tokens. Each character the user types becomes a `<CHAR_<UPPER>>` token, and the model is trained to predict the corrected word as plain text between `<XBC>` and `<XEC>`.

Right (verified, top-1 ~74% accuracy on the reference English model):
```
prompt:  <XBU><CHAR_T><CHAR_E><CHAR_H><XBC>
output:  The <XEC>...
```

Wrong (0% accuracy, model produces nonsense like "I" then `<XEC>` and gives up):
```
prompt:  <XBU>teh<XBC>
```

#### Implication for languages with diacritics

The Android keyboard sends one `<CHAR_X>` token per **physical key press**. Diacritics are not separate tokens. For Portuguese, this means:

- `ã` typed as long-press-`a` → emits `<CHAR_A>` only (the diacritic is NOT preserved in the prompt)
- `ç` → `<CHAR_C>`
- `é` → `<CHAR_E>`

Concretely, the typo-string-to-keypress conversion is:

```python
import unicodedata

def to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        decomposed = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        if base in ("ç", "Ç"):
            base = "C"
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out
```

This means *missing accents* is the dominant typo class for diacritic-rich languages — the model has to recover them every time. That's exactly what FUTO's autocorrect is designed for.

For languages outside Latin-26 (Cyrillic, Greek, Arabic, CJK), you would need to extend the `<CHAR_*>` set. This is risky because the C++ assumes a 26-letter Latin alphabet (`LETTERS_TO_IDS[0..25]`). Adding more `<CHAR_*>` tokens won't be picked up; the keypress-to-token mapping in the Android keyboard would need to be patched. This is out of scope for this guide.

### 3.4 GGUF metadata fields

The `keyboardlm.*` namespace in the GGUF holds FUTO-specific metadata. Reproduce these exactly:

| Field | Type | Required value pattern |
|---|---|---|
| `keyboardlm.languages` | STRING | e.g. `"pt-BR"` (BCP-47) |
| `keyboardlm.finetuning_count` | UINT32 | `0` for a fresh model |
| `keyboardlm.history` | STRING | freeform, e.g. `"2026-04-29: Model created"` |
| `keyboardlm.features` | STRING | space-separated feature flags (see below) |
| `keyboardlm.ext_tokenizer_type` | STRING | `"sentencepiece"` |
| `keyboardlm.ext_tokenizer_data` | **`[UINT8]`** | the raw bytes of your `.spm` file (see Gotcha 2 below) |

**The `features` string is critical.** The minimum that works for autocorrect is:

```
base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1
```

The wiki documents the first three but **not the fourth**. Without `char_embed_mixing_v1`, the C++ at `LanguageModel.cpp:94` skips populating `LlamaAdapter::embeddings`, and `DecodePromptAndMixes` (which runs on every keypress) does

```cpp
float *src = llamaAdapter->embeddings.data() + (t.token * n_embd);
for (size_t i = 0; i < n_embd; i++) mix_f[i] += src[i] * weight;  // SIGSEGV
```

against an empty vector. We discovered this by pulling the crash backtrace via wireless ADB — see [Phase 11](#11-side-loading-and-reproducing-a-crash). In practice you should always declare `char_embed_mixing_v1` if you declare `xbu_char_autocorrect_v1`. The two features are not actually independent in the current FUTO build, regardless of how the wiki frames them.

The reference English model also declares `xc0_swipe_typing_v1` (swipe input ML) and would require an extra encoder tensor. You can omit both safely; the keyboard falls back to dictionary-based swipe.

---

## 4. Pipeline overview

```
                             [Phase 5 — package & side-load]
                                          ▲
                                          │
[corpus] -- Phase 2 --> [tokenizer] -- Phase 3 --> [pretrain base]
                                                           │
                                                  Phase 4 (3 stages)
                                                           │
                                                   [final HF ckpt]
                                                           │
                                                   convert + patch
                                                           │
                                                   [.gguf v2 file]
                                                           │
                                                       phone
```

Each phase has its own script in this project at `scripts/0N_*.py`. Run them in order; output of phase N is input to phase N+1.

---

## 5. Phase 0 — extract the official model as your spec

Before you do anything else, get the reference and dump it:

```bash
mkdir my-keyboard && cd my-keyboard
python -m venv env && source env/bin/activate
pip install sentencepiece huggingface_hub gguf protobuf numpy

git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

hf download breadlicker45/futo-keyboard-lm --local-dir reference_model/
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
       reference_model/ml4_1_f16_meta_fixed.gguf > reference_metadata.txt
```

Treat `reference_metadata.txt` as your spec. Every metadata field, every architectural constant, every tensor name — match it.

You also want the embedded SentencePiece extracted from inside the GGUF, because it tells you which special-symbol strings actually appear:

```python
from gguf import GGUFReader
r = GGUFReader("reference_model/ml4_1_f16_meta_fixed.gguf")
spm_field = r.fields["keyboardlm.ext_tokenizer_data"]
spm_bytes = bytes(int(spm_field.parts[i].tolist()[0]) for i in spm_field.data)
open("reference_model/extracted_spm.model", "wb").write(spm_bytes)
```

Open `extracted_spm.model` with `sentencepiece` and print pieces 4..303 — that's the slot map.

---

## 6. Phase 1 — assemble a target-language corpus

For a phone keyboard, **register matters more than volume**. A 5B-token corpus of formal Wikipedia-style text gives you a model that's awful at predicting `vc tô indo agora` but great at predicting `the diaspora demographically referred to`. People type the first kind, not the second.

The mix that worked in our pt-BR run:

| Source style | HF dataset (April 2026) | Notes |
|---|---|---|
| Web crawl, target-language | (varies — `eduagarcia/BrWac` for pt-BR) | Bulk; broad statistics |
| Wikipedia | `wikimedia/wikipedia` config `<dateid>.<lang>` | Formal grounding |
| **Subtitles / film dialogue** | `Helsinki-NLP/opus-100` config `<en-yourlang>` | **The strongest casual-register signal** |

We tried `mozilla-foundation/common_voice_*` and `gabrielrstan/CORAA-v1.1` for additional conversational data. **Both failed in early 2026**: Common Voice requires auth or was restructured, and CORAA's data layout doesn't match HF's auto-loader. If you're reading this later, retry — they may be back. If not, scrape your own (forum/Reddit/Twitter dumps).

Streaming from HF is non-negotiable at this scale. Don't try to download the whole BrWaC or OSCAR — you'll run out of disk. Stream, filter, write sharded text files. The script `scripts/01_build_corpus.py` shows the pattern; key points:

- `streaming=True` to `load_dataset`
- per-doc filter: minimum length (lower for subtitles), drop noise, language-filter for mixed-locale corpora
- dedup by hash of first ~200 chars
- write to `shard_NNNNN.txt` files of ~256 MiB each
- track approximate token count and stop at a budget

For a mini run, **500M tokens is enough** to validate the pipeline. For a real run, target **3-5B tokens** mixed across registers.

---

## 7. Phase 2 — train the SentencePiece tokenizer

You're training a 15008-vocab BPE with the 300 user-defined symbols pinned at indices 4..303. The minimal training call:

```python
import sentencepiece as spm

USER_DEFINED = build_300_symbols()  # see project script for the slot layout

spm.SentencePieceTrainer.train(
    input=",".join(corpus_shards),
    input_format="text",
    model_prefix="tokenizer/spm_<lang>",
    vocab_size=15008,
    character_coverage=0.9995,
    model_type="bpe",
    treat_whitespace_as_suffix=True,   # the "inverted_space" feature
    user_defined_symbols=USER_DEFINED, # gets IDs 4..303 in declaration order
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    byte_fallback=True,
    input_sentence_size=2_000_000,
    shuffle_input_sentence=True,
)
```

Validate immediately:

```python
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm_<lang>.model")
assert sp.get_piece_size() == 15008
char_ids = [sp.piece_to_id(f"<CHAR_{c}>") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
assert char_ids == list(range(char_ids[0], char_ids[0]+26)), \
    "<CHAR_A>..<CHAR_Z> must be sequential"
assert sp.piece_to_id("<XBU>") > 3
```

If the assertion fails on `<CHAR_*>`, your `user_defined_symbols` list isn't sequential. Fix the order.

### The big RAM trap

SentencePiece BPE is memory-bound, not compute-bound. The merge phase keeps the entire training set + a pair-counts hash table resident, and chases random pointers through it. It's roughly **5-10x the input size in RAM**.

In our run, training on a 2.0 GiB corpus consumed 10.6 GiB of RAM, used a single CPU core at ~10%, and ran for **5+ hours on a 16 GiB host that started swapping to disk**. The same training on a 31 GiB host with DDR4-4000 RAM (no swap pressure) finished in **14 minutes**. Same algorithm. Same data. The only difference was memory headroom.

If your training host has < 32 GiB RAM, do one of:

1. Pre-sample your corpus to ~1-2 GiB before training: `shuf shards/*.txt | head -c 1500000000 > sp_sample.txt`
2. Run tokenizer training on a different machine with more RAM and copy the `.spm` over.
3. Add 32+ GiB of swap as a safety net (slow but won't OOM-kill).

This is the kind of thing nobody tells you upfront. We almost killed a run because we thought the algorithm had hung.

---

## 8. Phase 3 — pretrain the base model

This is a standard HuggingFace Trainer loop with the verified `LlamaConfig`. Key decisions:

```python
TrainingArguments(
    max_steps=100_000,                  # for a real run; 20k for mini validation
    per_device_train_batch_size=16,     # micro batch
    gradient_accumulation_steps=16,     # → global batch 256
    learning_rate=3e-4,
    warmup_steps=2000,
    lr_scheduler_type="cosine",
    bf16=True,
    weight_decay=0.1,
    save_steps=5000,
)
```

Sequence length of 1024 fits comfortably in 24 GiB at micro-batch 16. With seq=1024 × 256 batch = ~262k tokens per step. 100k steps × 262k = ~26B training tokens. On a 5B-token corpus that's about 5 epochs.

Stream the corpus rather than loading it. Tokenize on the fly and pack to fixed length. Workers are I/O-bound, not GPU-bound — 2-4 workers is fine.

**Save every 2500-5000 steps** so you can resume from any checkpoint without losing more than a few hours.

Expect a final cross-entropy loss around 3-4 on a real-quality run, perplexity around 30-50. The English reference probably had perplexity around 25-30. You'll see your loss plateau by 60-80% of total steps; the rest is fine-tuning the long-tail vocabulary.

### Mini-run reality check

We did a 20k-step pretrain on 500M tokens of Wikipedia-pt only. Final loss 3.81, perplexity ~45. Loaded fine, ran fine, but suggestions were poor. Don't expect a 20k-step Wikipedia-only run to produce a usable keyboard model — it's a smoke test, nothing more.

---

## 9. Phase 4 — autocorrect fine-tune (3 stages)

The pretrain model has no idea what the `<XBU>...<XEC>` format means. Phase 4 teaches it. Three sequential stages, each loading from the previous:

### Phase 4a: isolated autocorrect triples

Generate a synthetic dataset of `<XBU><CHAR_*>...<XBC>correct<XEC>` examples, sampling words from a frequency map of your corpus. Strategies for the typo-string side:

- Drop accents (the dominant typo class for diacritic languages: `voce`→`você`, `nao`→`não`)
- Cedilla loss: `coracao`→`coração`
- Keyboard adjacency typos (qwerty layout)
- Transposed/doubled/missing characters
- Common shortcuts your speakers actually type (`vc`→`você`, `tb`→`também` in pt-BR)

Weight the sampling by `log(word_freq + 1)` so common words don't drown out rare ones (that's what the FUTO wiki recommends, and it's reasonable).

Train with **labels masked everywhere except the correction span** (between `<XBC>` and `<XEC>`):

```python
labels = [-100] * len(input_ids)
xbc_pos = input_ids.index(sp.piece_to_id("<XBC>"))
for k in range(xbc_pos, len(input_ids)):
    labels[k] = input_ids[k]
```

This focuses the loss on what the model needs to actually *predict*. About 5-10K steps with seq_len=64, batch 256, lr 1e-4 is enough — going longer overfits the synthetic distribution.

### Phase 4b: in-context autocorrect

Take real sentences from your corpus, randomly replace ~20-30% of words with `<XBU><CHAR_*>...<XBC>correct<XEC>` triples in place. Loss now computes over all tokens (no masking), so the model learns both autocorrect and continued language modeling.

```
Eu fui ao <XBU><CHAR_M><CHAR_E><CHAR_C><CHAR_A><CHAR_D><CHAR_O><XBC>mercado<XEC> ontem
```

Train for 20-30K steps at seq_len=512, lr 5e-5. Be careful with the typo rate — too high (>40%) and the model mode-collapses to "always emit a triple". We saw this with rate 0.33 in our run; 0.20-0.25 is safer.

### Phase 4c: conversational adaptation

This stage is **not** in the FUTO wiki. We added it after observing that even Phase 4b's output sounded like Wikipedia. Take a smaller corpus of casual text in your language (subtitles, chat, etc., not formal web crawl), apply Phase 4b's typo-injection at a *lower* rate (~0.10), and do a short fine-tune (~5K steps, lr 2e-5). This shifts the model toward typing-style register without erasing the language statistics.

---

## 10. Phase 5 — package as a FUTO-compatible GGUF

This phase is short, frustrating, and where the most non-obvious gotchas live. Five things must be right for the file to load and not crash:

1. **GGUF version 2** (FUTO's vendored llama.cpp can't read v3+).
2. **No KV fields the older parser doesn't recognize** (~9 specific fields produced by recent `convert_hf_to_gguf.py` must be stripped).
3. **`output.weight` quantized to Q6_K** (matches reference; F16 may work but matching reference is safer).
4. **`keyboardlm.features` includes `char_embed_mixing_v1`** (otherwise SIGSEGV mid-inference, see Section 11).
5. **`keyboardlm.ext_tokenizer_data` is a `[UINT8]` array, not `[INT32]`.**

Concrete pipeline:

```bash
# 5.1: stage the HF checkpoint with the SentencePiece as tokenizer.model
mkdir -p staged
cp finetune/stage_c/final/* staged/
cp tokenizer/spm_<lang>.model staged/tokenizer.model
echo '{"tokenizer_class":"LlamaTokenizer","model_max_length":2048,...}' \
     > staged/tokenizer_config.json
echo '{"bos_token":"<s>","eos_token":"</s>","pad_token":"<pad>","unk_token":"<unk>"}' \
     > staged/special_tokens_map.json

# 5.2: HF -> vanilla GGUF via llama.cpp
python llama.cpp/convert_hf_to_gguf.py staged/ \
       --outfile vanilla.gguf --outtype f16

# 5.3: requantize output.weight to Q6_K (matches reference)
./llama.cpp/build/bin/llama-quantize \
    --allow-requantize \
    --output-tensor-type q6_k \
    vanilla.gguf q6kout.gguf f16

# 5.4: patch FUTO metadata into the GGUF
python scripts/06_patch_metadata.py \
    --in q6kout.gguf --out futo_v3.gguf \
    --tokenizer tokenizer/spm_<lang>.model \
    --languages "<lang-tag>" \
    --features "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"

# 5.5: downgrade GGUF v3 → v2 and strip extra fields
python scripts/06b_downgrade_v2.py --in futo_v3.gguf --out futo_v2_final.gguf
```

The `06_patch_metadata.py` step has one subtle requirement: **embedding the SentencePiece bytes**. The straightforward `writer.add_array(name, list(spm_bytes))` produces a `[INT32]` array, which the FUTO C++ rejects. Use `add_key_value` with explicit `sub_type`:

```python
from gguf import GGUFValueType
spm_bytes = open("tokenizer.model", "rb").read()
writer.add_key_value(
    "keyboardlm.ext_tokenizer_data",
    spm_bytes,                         # bytes, not list[int]
    GGUFValueType.ARRAY,
    sub_type=GGUFValueType.UINT8,      # critical
)
```

The downgrade step (`06b_downgrade_v2.py`) reads the GGUF, copies all fields except a deny-list of 9 newer-than-v2 fields (`general.size_label`, `general.type`, `llama.attention.key_length`, `llama.attention.value_length`, `llama.vocab_size`, `tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token`, `tokenizer.ggml.padding_token_id`, `tokenizer.ggml.pre`), writes a fresh GGUF, and patches the version byte at offset 4-7 from `\x03\x00\x00\x00` to `\x02\x00\x00\x00`.

After all of this, your `futo_v2_final.gguf` should be ~62 MB and have exactly **28 KV fields** matching the reference English model's layout. Do a final diff:

```bash
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py futo_v2_final.gguf > ours.txt
diff <(grep -oP "(?<=\| )[a-z._]+(?= = )" reference_metadata.txt | sort) \
     <(grep -oP "(?<=\| )[a-z._]+(?= = )" ours.txt | sort)
```

The only differences should be the `keyboardlm.languages` and `keyboardlm.history` strings (your language vs `'en'`, your date vs `'2023-11-11'`).

---

## 11. Side-loading and reproducing a crash

Transfer the `.gguf` to the device — Syncthing, Nextcloud, USB, or `adb push /sdcard/Download/`. Then in FUTO Keyboard: **Languages & Models → Add Model → select your file → assign to your language → Text Prediction → enable Transformer LM**.

If it loads as "(Unsupported)": open the GGUF dump and check that all six `keyboardlm.*` fields are present and the version is 2.

If the keyboard opens normally but **closes the moment you type 2-3 letters**: it's a native crash (SIGSEGV), almost always one of:

- Missing `char_embed_mixing_v1` feature
- `keyboardlm.ext_tokenizer_data` is INT32 instead of UINT8
- One of the structural special tokens (`<XBU>`/`<XBC>`/`<XEC>`/`<XC0>`/`<CHAR_A>`) didn't make it into your SentencePiece

To debug, enable wireless ADB on the phone (Developer Options → Wireless Debugging → Pair device with code), then from your Linux box:

```bash
adb pair <IP>:<PAIRING_PORT> <CODE>     # one-time
adb connect <IP>:<DEBUG_PORT>            # each session
adb logcat -c                            # clear buffer
adb logcat | grep -E "Fatal signal|F libc|F DEBUG|appDiedLocked.*futo"
```

Reproduce the crash. You'll see something like:

```
F libc    : Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x64800
            in tid NNN (LanguageModel)
F DEBUG   : signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000000000064800
F DEBUG   : backtrace:
F DEBUG   :   #00 pc 0x1681ac libjni_latinime.so
              (LanguageModelState::DecodePromptAndMixes(...) + 1904)
```

A small fault address (under 0x100000) means a `nullptr + offset` dereference, almost always a missing feature or uninitialized field. The function name and offset point at exactly which member is uninitialized — read `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` around the offset, and `native/jni/src/ggml/LanguageModel.cpp` for the `LlamaAdapter` initialization.

If you see `ASSERT failed` early at app start (before you type), you have a missing structural token in the tokenizer. The asserts are at `LanguageModel.cpp:222-225` and 233.

---

## 12. The five gotchas, in one place

In order from "the most likely to bite a new person" to least:

1. **The keypress format is `<CHAR_X>` tokens, not literal text.** Wiki snippet hints at it but is easy to misread. Verifying via a test prompt against the reference English model takes 30 seconds and saves days.
2. **`char_embed_mixing_v1` is required if you declare `xbu_char_autocorrect_v1`.** Wiki frames them as independent features. They're not. Without `char_embed_mixing_v1`, every keystroke SIGSEGVs.
3. **GGUF must be version 2.** Newer `convert_hf_to_gguf.py` produces v3 with extra KV fields the FUTO-vendored llama.cpp can't parse. You need a downgrade pass.
4. **`keyboardlm.ext_tokenizer_data` must be `[UINT8]`.** The natural-looking `add_array(name, list(bytes))` produces `[INT32]` (4× the size, wrong layout). Use `add_key_value(..., sub_type=GGUFValueType.UINT8)`.
5. **`<CHAR_A>..<CHAR_Z>` must be 26 contiguous, sequential token IDs.** The C++ does pointer arithmetic on `<CHAR_A>`'s ID. Listing them in order in `user_defined_symbols` is sufficient.

There are smaller things too — `<CHAR_A>=182` and `<XBU>=174` in the reference (we kept the exact same indices for safety even though most are name-resolved); `output.weight` is Q6_K rather than F16 in the reference; the SentencePiece in the GGUF is the *real* tokenizer that the C++ uses, not the `tokenizer.ggml.tokens` array. None of these is fatal individually, but together they're enough to lose a weekend if you don't know to check.

---

## 13. Evaluation methodology

Before rolling onto a phone, evaluate against the same test suite at three checkpoints:

| Reference | What it represents |
|---|---|
| **Ceiling** — official English model run on English tests | Best-case quality with this architecture and data scale |
| **Floor** — your pretrain checkpoint *before* Phase 4 | Tests autocorrect on a model that hasn't seen the format. Should be near 0%. |
| **Final** — your post-Phase-4 model | Should be substantially above floor; ideally close to ceiling. |

For autocorrect tests, use ~30 examples per language across categories: shortcuts, missing-accent, transposed, doubled, common misspellings. For each: feed `<XBU><CHAR_*>...<XBC>` and check if the model emits the correct word before `<XEC>`. Track top-1 (greedy) and top-5 (sampled).

The English reference scores ~74% top-1 / 89% top-5 on a 27-question English autocorrect suite of the kind described above. A successful pt-BR-quality run should aim for similar numbers on a parallel pt-BR suite.

A failure mode to watch for: **final ≈ floor**. That means Phase 4 didn't teach the model the format. Check (a) that the dataset actually uses the keypress format, (b) that label masking in Phase 4a is correct, (c) that loss decreased substantially during Phase 4. If your final loss after Phase 4a is >1.0, fine-tuning probably didn't converge.

---

## 14. What this guide does not cover

- **Hyperparameter tuning past a working baseline.** The configs above produce *a* working model; getting it to ceiling-level quality is its own multi-week effort.
- **Quantization beyond Q6_K output.** Reference is mostly F16 with Q6_K output; smaller quantizations (Q4_K_M etc.) reduce file size but we haven't tested compatibility with FUTO's loader.
- **On-device LoRA fine-tuning.** The `lora_finetunable_v1` feature exists in FUTO's spec but we didn't enable it. Adding it requires preparing the model with specific tensor metadata; out of scope here.
- **Swipe-typing ML (`xc0_swipe_typing_v1` + `experiment_linear_208_209_210`).** Requires additional encoder tensors at hard-coded indices 208/209/210. Skip it; FUTO's classical swipe still works without ML and the keyboard remains usable.
- **Languages outside Latin-26.** The 26 hardcoded `<CHAR_*>` slots assume an alphabet of 26 characters. Cyrillic, Greek, Arabic, etc. would need patches in the FUTO Android keyboard itself — not just a different tokenizer.
- **Production deployment.** This is a side-loading guide. Distributing your model as an installable package or contributing it back to FUTO upstream is a different conversation.

---

## 15. Acknowledgements and licensing

This guide is independently written. It documents undocumented behavior of the FUTO Keyboard Android app (open source, see the FUTO repo for its license) by reverse-engineering crashes, reading the C++ source, and verifying behavior via real inference. The architectural and metadata details are facts about how the app works; they are not creative work owned by FUTO.

The reference English model `breadlicker45/futo-keyboard-lm` is a re-upload of FUTO's official English keyboard model. We use its byte-level structure as a specification target; we do not redistribute it.

Concrete project artifacts (training scripts, GGUF metadata patcher, downgrade script) are licensed under MIT — adapt them freely for your language. The model weights you produce are yours.

If FUTO publishes official multi-language training scripts, prefer those over this guide. They've said they intend to revisit the ML pipeline; if you do this work, your model may need re-packaging when that happens. Treat any model you ship as a "good for the next 6-18 months" investment, not a permanent solution.

The fastest way to validate FUTO's wiki claims is the same way we did: train, package, side-load, and read the crash. Several discoveries in this guide came from a SIGSEGV at offset 0x64800 in `DecodePromptAndMixes` and the subsequent half-day of source diving. The format spec emerges from doing the work — not from reading.

Have fun, and good luck.
