<p align="right"><strong>🇺🇸 English</strong> · <a href="README.pt-BR.md">🇧🇷 Português</a></p>

# FUTO Keyboard — Brazilian Portuguese language model

A 36M-parameter Llama-architecture transformer that ships inside [FUTO Keyboard](https://keyboard.futo.org) and gives you private, **on-device** autocorrect and next-word prediction in **Brazilian Portuguese**.

FUTO only ships an English model. Their stated position is "we're working on others" with no ETA, and they explicitly support side-loading third-party models in the same format — but the format spec on the public wiki is incomplete and several integration steps are non-obvious. This repo closes those gaps **and** publishes the trained pt-BR model.

> **Latest release:** [`v8.2`](../../releases/latest) — Q6_K GGUF, 62 MB, drops into FUTO and runs on a real Android phone.

---

## TL;DR

| | |
|---|---|
| Architecture | Llama, 8 layers, 512 hidden, 36M params (matches FUTO's English reference) |
| Tokenizer | SentencePiece BPE, 15008 vocab, 300 fixed user-defined slots |
| Pretrain corpus | ~4B pt-BR tokens (BrWaC, OSCAR, Wikipedia, Carolina, OpenSubtitles, CORAA) |
| Hardware to reproduce | 1× RTX 3090 (24 GiB) or any 16+ GiB consumer GPU |
| Wall-clock to reproduce | ~30–50h pretrain + ~3.5h fine-tune + <30min packaging |
| Real-typo eval (50-pair holdout) | **60% top-1 / 72% top-5** (v8.2) |
| Best category | **accent_only: 88% top-5** (the dominant pt-BR error class) |
| Weak spots (still) | hybrid multi-error, prefix completion |

The whole training pipeline, the format spec we reverse-engineered, and the trained GGUF are MIT-licensed. **You should be able to do this for your language in a weekend on a single consumer GPU.**

---

## Why this exists

A keyboard you actually type into all day is the single best place for a small private LM. FUTO Keyboard already runs one on-device via [llama.cpp](https://github.com/ggerganov/llama.cpp) — but only in English, and the official wiki entry on the format is incomplete enough that several non-trivial things have to be reverse-engineered from the binary GGUF and a crash backtrace.

This repo is the record of doing that for Brazilian Portuguese end-to-end. It's also a **path-of-stones** for anyone who wants to do it for *their* language: every gotcha we hit, every dead end we ruled out, every loss-curve number, in version-by-version order.

---

## The journey

Every version targets a real number on a real-typo holdout (typed on a real phone, then corrected by the same human). Categories are what they sound like: `accent_only` (cafe → café), `adjacency_or_short_edit` (single-key slip), `prefix_completion` (typed first 2–3 letters, expect the rest), `hybrid` (multi-error), `cedilla_only` (c → ç).

### v2 — the wall (~6h, May 2026)

Built the whole pipeline end to end. Followed the FUTO public wiki as written.

| Stage | top-1 | top-5 |
|---|---|---|
| stage_a | 1/50 (2%) | 6/50 (12%) |
| stage_b | **0/50** | **0/50** |
| stage_c | **0/50** | **0/50** |

Mode collapse: the model learned to always emit the literal corrected text it was being trained on, ignoring the keypress prompt. Five hyperparameters had been changed at once — none could be blamed individually.

**Lesson:** *don't change five things at once; don't trust an incomplete recipe.*

### v3 — the diagnosis (research + ablations, ~1 day, 2026-05-12)

Four rounds of literature ([SwiftKey/Gboard/Apple papers](notes/v3_research_synthesis.md), fine-tuning theory, prompt-loss masking, pt-BR specifics) plus five small **single-variable** ablations on a 5070 Ti / 3090 split. Two real causes fell out:

1. **The 04b/04c loss formulation was mechanically wrong.** It computed loss over the *full sequence* (PLW=1.0). ~90% of the gradient came from clean pt-BR tokens; the 10% of XBU autocorrect tokens were drowned out. Fix: PLW=0.05 (per [arxiv:2401.13586](https://arxiv.org/abs/2401.13586)), which scales clean-token loss down without zeroing it. Also: an off-by-one bug shifting labels one token early.
2. **SAM (Sharpness-Aware Minimization) hurt 4a.** Tried because of catastrophic forgetting; ablated, regressed every checkpoint vs vanilla AdamW, rejected.

| Ablation | top-1 | top-5 | Verdict |
|---|---|---|---|
| A1 (PLW=0.0) | 3/50 (6%) | 6/50 (12%) peak, overfits | baseline collapse |
| A2 (PLW=0.05) | 4/50 (8%) | 8/50 (16%) | **fix confirmed** |
| A5 (PLW=0.05 + SAM) | 0 | 0–1 | SAM rejected |
| **B1** (4b, PLW=0.05) | 8/50 (16%) | **15/50 (30%)** at step 4000 | **ship gate cleared** |

Stop-go criterion (≥30% top-5) passed. The 200K-pair hand-rolled synth pool was right-sized; the Claude-API generation we'd planned was deferred. **Time to ship a real v8.**

### v4–v7 — the packaging gauntlet and the false ship (Apr 29 → May 3, 2026)

Between the first pipeline working end-to-end and the diagnosis on May 12, the model was repackaged half a dozen times and shipped to a real phone once. Each iteration closed one gap in the GGUF format spec — and exactly one of them turned out to be the model itself, not the format. (The "v3" label here is the early *packaging* tag, distinct from the May 12 *research* "v3 (the diagnosis)" above — the label got reused once the recipe story took over.)

| Tag | When | What it actually was |
|---|---|---|
| `v3` (packaging) | Apr 29 11:35 | third packaging fix for the mini-corpus run — got the [tokenizer slot layout](GUIDE.md#32-the-tokenizer-300-user-defined-symbols-at-fixed-slots) right |
| `v4` | Apr 29 11:49 | fourth packaging fix — got the [GGUF metadata fields](GUIDE.md#34-gguf-metadata-fields) FUTO actually reads |
| `v5` | Apr 29 12:24 | fifth fix + a casual-corpus variant — survived [Q6_K output quantization](GUIDE.md#12-the-five-gotchas-in-one-place) without `SIGSEGV` on the second keystroke |
| `v6` | May 3 19:58 | first big-corpus model packaged correctly enough to load on a real phone. **Real-typo eval: 0/50 top-5.** Predictions looked like noise. |
| `v7` | (planned, never shipped) | the `finetune_big_v2/` retrain — same broken Phase 4 recipe, fresh weights. `scripts/run_phase4_v2.sh:171` still carries the comment: *"Next: pull finetune_big_v2/stage_c/final/ → build GGUF v7 → side-load"*. `eval_real_v2_stage_c.json` (also 0/50 top-5) is what v7 would have shipped with. |

**The jump from v6 directly to v8** (skipping v7) marks the moment we stopped iterating on packaging and started questioning the recipe. v7 was almost shipped before the brake got pulled in favor of [the v3 research synthesis](notes/v3_research_synthesis.md).

**Lessons** (these are the source of [GUIDE.md §11](GUIDE.md#11-side-loading-and-reproducing-a-crash) and [§12 the five gotchas](GUIDE.md#12-the-five-gotchas-in-one-place)):

- **Packaging bugs and training bugs look identical on a phone** — both produce noise. Smoke-test your packaging pipeline against a model you *know* works (the English reference GGUF) before trusting it to evaluate yours.
- **Format gotchas are sequential.** Fixing one unmasks the next: we didn't discover the Q6_K-output `SIGSEGV` until *after* the metadata fields were right, because before that the model wouldn't even load. Plan for at least three packaging passes when adapting to a new format.
- **Never ship a model that hasn't passed a real-typo holdout in Python first.** v6 went to a real phone with only synthetic-typo eval. The on-device 0% was the wake-up call, and it took **9 days** to figure out the recipe was the cause, not the bytes.
- **When you've repackaged three times and the model still doesn't predict, the problem is upstream of packaging.** Stop iterating on the wrong layer and go look at the loss formulation.

### v8 — the first ship (~4h on 3090, 2026-05-12)

Full Phase 4 (a + b + c) with PLW=0.05, off-by-one fix, no SAM. Real-typo mix 25%, 200K synth + 343 real pairs. Packaged as Q6_K-`output.weight` + F16 GGUF v2 (matches the English reference exactly).

| Category | n | top-1 | top-5 |
|---|---|---|---|
| **Overall** | 50 | **36.0%** | **56.0%** |
| accent_only | 17 | 52.9% | **88.2%** |
| adjacency_or_short_edit | 21 | 42.9% | 61.9% |
| hybrid | 5 | 0% | 0% |
| prefix_completion | 7 | 0% | 0% |

Excluding the categorically-hard buckets (hybrid + prefix), v8 hit **74% top-5 on solvable pairs**. The `v8.gguf` loads in FUTO Keyboard 0.1.27 on Android and predicts. Shipped as the first real artifact.

**Lesson:** *fix the mechanical bug, the rest of the recipe was basically fine.*

### v8.1 — the sharpening attempt (~5h, 2026-05-22) — NOT shipped

Two changes targeting top-1 sharpness and the weak categories:

- **Refreshed real-typo pool** from 343 → 393 unique pairs after dedup; carved a strictly-disjoint **58-pair v8.1 harder holdout**, intentionally over-weighted on prefix and hybrid.
- **Claude-CLI synthetic pairs** (`claude -p` shell-out, no API key) — 500 general + 250 weak-spot-targeted pairs, generated using the two-stage [Google Gemini](https://research.google/blog/improving-mobile-keyboard-language-models/) prompt pattern.
- **PLW_C = 0.02** in stage_c to sharpen top-1.

Result on the v8 holdout: **44% top-1 / 64% top-5** (+8pp / +8pp over v8). On the harder v8.1 holdout: 39.7% top-1 / 55.2% top-5.

But **NWP regressed**: conversational next-word top-3 went from v8's 6.1% to v8.1's 0.7% (masked). The sharper recipe killed free-form context. **Not shipped.**

**Lesson:** *don't optimize one metric without watching the others.* Saved as a `feedback` memory so this doesn't happen again.

### v8.2 — the better recipe (~3.5h fine-tune, 2026-05-22 → 23) — **current release**

Stepped back from PLW_C=0.02. Kept the Claude weak-spot pool. Added a **continue-pretrain pass on a conversational corpus** (~42K message-style snippets) before fine-tuning, to give the base model a stronger casual-register prior.

Phase 4 wall-clock on a single 3090:

| Stage | Steps | Wall-clock |
|---|---|---|
| 4a (isolated autocorrect) | 3000 | **7m 04s** |
| 4b (in-context autocorrect) | 12000 | **2h 40m 51s** |
| 4c (conversational adaptation) | 10000 | **48m 56s** |

Result on the same 50-pair v8 holdout:

| Category | n | top-1 | top-5 |
|---|---|---|---|
| **Overall** | 50 | **60.0%** | **72.0%** |
| accent_only | 17 | 82.4% | 88.2% |
| adjacency_or_short_edit | 21 | 66.7% | 81.0% |
| hybrid | 5 | 20% | 40% |
| prefix_completion | 7 | 14.3% | 28.6% |

On the harder v8.1 holdout: 39.7% top-1 / **63.8% top-5** (vs v8.1's 55.2%) — +8.6pp top-5 with the same top-1 number, on a holdout that intentionally over-weights the hard categories. Masked NWP top-3 is 4.7% (still below v8's 6.1% — honest regression, real conversational signal got partially sacrificed for typo accuracy; we're tracking this).

This is the **featured release**. The v8 GGUF stays available for users who want the original proven artifact.

**Lesson:** *measure on multiple holdouts and free-form NWP, not just one typo set.*

---

## At a glance — version comparison

| Version | Top-1 (v8 holdout) | Top-5 (v8 holdout) | NWP top-3 (masked) | Shipped? |
|---|---|---|---|---|
| v2 stage_a | 2% | 12% | — | no (collapsed) |
| v2 stage_b/c | 0% | 0% | — | no (collapsed) |
| **v8** | 36% | 56% | **6.13%** | **yes (legacy)** |
| v8.1 | 44% | 64% | 0.75% | no (NWP regression) |
| **v8.2** | **60%** | **72%** | 4.66% | **yes (latest)** |

---

## Reproduce it for your language

Full step-by-step technical reference: [**GUIDE.md**](GUIDE.md) (English) · [**GUIDE.pt-BR.md**](GUIDE.pt-BR.md) (Português). What it actually takes:

1. **Phase 0** — extract FUTO's English reference GGUF as your spec (`scripts/00_inspect_reference.py`). The format is largely undocumented; the binary is authoritative.
2. **Phase 1** — assemble a 3–5B-token corpus in your target language (`scripts/01_build_corpus.py`, streams from HuggingFace). Strongest casual-register signal we found: OpenSubtitles.
3. **Phase 2** — train a SentencePiece BPE tokenizer, **15008 vocab**, with the 300 user-defined symbols pinned at IDs 4..303 in *exactly* the FUTO order (`scripts/02_train_tokenizer.py`).
4. **Phase 3** — pretrain (`scripts/03_pretrain.py`, ~100K steps, 30–50h on a 3090).
5. **Phase 4** — autocorrect fine-tune in three stages (`scripts/04a_*`, `04b_*`, `04c_*`). **Use PLW=0.05** in stages b and c — this is the one mechanical thing the wiki gets wrong by omission.
6. **Phase 5+6** — convert to GGUF, downgrade to GGUF v2, patch FUTO metadata, **Q6_K-quantize `output.weight`** (F16 output crashes FUTO's JNI on the second keystroke). Scripts: `05_to_futo_gguf.py`, `06_patch_metadata.py`, `06b_downgrade_v2.py`.
7. **Smoke test on a real Android phone before celebrating.** Python eval ≠ on-device eval (different quant, different feature flags, different sampler).

Budget: **one weekend on a 24 GiB consumer GPU**, plus a few hours of Python familiarity. Hardest single thing is the tokenizer slot layout — that's why `notes/reference_slot_map.md` exists.

---

## The five gotchas, in one line each

(Full discussion: [GUIDE.md §12](GUIDE.md#12-the-five-gotchas-in-one-place).)

1. **The prompt format is a keypress sequence (`<CHAR_X>` tokens), NOT literal text.** This is the load-bearing wrong claim on the public wiki.
2. **`<CHAR_A>..<CHAR_Z>` must be 26 contiguous token IDs.** FUTO's C++ does pointer arithmetic on them.
3. **GGUF must be v2.** llama.cpp writes v3 by default; FUTO's bundled llama.cpp doesn't read it.
4. **`output.weight` must be Q6_K.** F16 output → `SIGSEGV` on the second keystroke. Not in the wiki.
5. **PLW must be ~0.05 in Phase 4b/c.** Otherwise the model mode-collapses to always emitting the corrected text. Not in the wiki.

---

## Honest limitations

- Production keyboard transformers (SwiftKey's published numbers) gain ~**1 percentage point** on NWP over a GRU baseline. Don't expect miracles. The win is on autocorrect category coverage, not raw next-word top-1.
- Real-typo accuracy depends on *your* typing patterns — a 50-pair holdout is informative, not definitive. Build your own holdout from your own typo log.
- FUTO Keyboard versions **after 0.1.27** have an upstream regression in empty-prefix NWP suggestions — app-side bug, not a model bug. Diagnosis and workaround: [`notes/futo_regression.md`](notes/futo_regression.md).
- `prefix_completion` and `hybrid` (multi-error) remain the hardest categories. v8.2 only partially addresses them. Open problem.

---

## Install the model

1. Download `futo_pt_br_v8_2.gguf` from the [latest release](../../releases/latest).
2. Side-load via FUTO Keyboard settings → Models → Import. Full instructions and ADB walkthrough: [GUIDE.md §11](GUIDE.md#11-side-loading-and-reproducing-a-crash).
3. If anything crashes, send back the `adb logcat` and we'll have a look.

---

## Acknowledgements

- **FUTO** for shipping a private on-device keyboard and explicitly supporting third-party models.
- **llama.cpp** for being the runtime that makes this size of model viable on a phone.
- **HuggingFace** and the maintainers of BrWaC, OSCAR, Wikipedia-pt, Carolina, OpenSubtitles, Common Voice, and CORAA.
- **Papers**: arxiv:2401.13586 (Prompt Loss Weight), arxiv:2505.05648 (SwiftKey privacy-preserving transformers), Google's mobile keyboard LM blog post (two-stage synthetic typo generation).
- The FUTO Keyboard wiki, including the parts that are wrong — getting them wrong is what produced this guide.

## License

- All Python scripts, shell scripts, and prose in this repo (this README, `GUIDE.md`, `GUIDE.pt-BR.md`, files in `notes/`): **MIT**. See [`LICENSE`](LICENSE).
- The trained `.gguf` model weights are released under the **same MIT license** — feel free to use, redistribute, modify.
- Training data is **not** redistributed; only references to public datasets and the scripts that fetch them.
- FUTO Keyboard itself is under the FUTO Source First License 1.1 (separate project, not bundled here).
