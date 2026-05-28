# pt-BR FUTO LM — v3 Research Synthesis

**Date:** 2026-05-12
**State going in:** v2 big run complete. Stage_a hits **12% top-5** on 50-pair real typo hold-out. Stage_b/c collapse to **0%** (same mode collapse as v1, despite a 5-variable change bundle).

This document captures the entire investigation since v2 results landed: the 5 lessons-learned saved to memory, four rounds of literature research (Gboard/SwiftKey/Apple, fine-tuning theory, hobbyist experiments, pt-BR particularities), and the revised v3 plan.

---

## 1. Lessons learned from v1 + v2 (saved as memory)

1. **Don't trust FUTO wiki training recipes.** Wiki was wrong on tokenizer, output quant, feature flags. Verified directly: the wiki itself marks loss function, hyperparameters, step counts, typo generation as "TODO". We've been guessing inside an officially-incomplete document.
2. **04b/04c are missing the XBU loss mask** that 4a uses. Full-sequence loss means ~90% of gradient comes from clean pt-BR tokens. Mechanical cause of stage_b/c collapse.
3. **Eval cadence too coarse.** A 5h training run with a single endpoint eval produces no trajectory; we cannot localize the collapse step. Need eval-every-500-steps callback.
4. **Python eval ≠ on-device eval.** HF transformers + bf16 ≠ llama.cpp + Q6_K + FUTO's feature flags (`char_embed_mixing_v1` etc.). Smoke-test every checkpoint on phone.
5. **Small single-variable ablations beat multi-hour gambles.** v2 changed 5 variables at once over ~6h. Stage_a's 12% is uninterpretable.

---

## 2. Research findings (4 rounds)

### Round 1 — Big-keyboard players (Gboard, SwiftKey, Apple)

**Most directly applicable: [SwiftKey Privacy-Preserving Transformers, arxiv 2505.05648](https://arxiv.org/html/2505.05648) (May 2025)**

| Aspect | SwiftKey | Us |
|---|---|---|
| Layers | **4** | 8 |
| Hidden / embed | 512 / 128 | 512 / 512 |
| Vocab | 20K **word-level** | 15K subword |
| Total size | **6MB quantized** | ~30MB Q6_K |
| Pretrain data | 11.1B tokens | 4B tokens |
| Fine-tune data | **2.63B real typed tokens from 6.8M users (DP)** | 200K synthetic + 343 real |
| Fine-tune steps | 28K | 20K |
| Effective batch | 64K | 12K |
| LR | 1e-4 | 1e-4 |
| Grad clip | 0.01 | default |
| Format | **plain NWP** | XBU/XBC/XEC |

Three critical observations:
- **SwiftKey does NOT use any autocorrect-specific format.** Their fine-tune is plain next-word prediction on real typed text. Autocorrect is handled by their "Fluency" search engine combining the LM with other signals at inference. **We are paying significant complexity for the XBU format with no evidence it's required.**
- They use **real user data** at 6.8M-user scale; we have ~300 real pairs. Different paradigm.
- **Production transformer barely beats their GRU baseline**: 90.83% vs 90.90% accuracy, 17.34 vs 16.20 NWP. Verbatim: *"We get small and consistent gains... gains appear primarily for new users before dynamic models activate."*

**[Google "Synthetic and federated" research blog](https://research.google/blog/synthetic-and-federated-privacy-preserving-domain-adaptation-with-llms-for-mobile-applications/) — current Gboard method (2024)**

Two-stage Gemini prompting to generate {clean → typo} pairs:
> 1. *"Here are some common grammatical errors: …Now apply these grammatical errors to the original sentences, and generate the ungrammatical sentences."*
> 2. *"Finally, correct the grammatical errors in the generated ungrammatical sentences. Do not modify the sentences except correcting the grammatical errors."*

Reported result: **22.8% relative NWP improvement** vs web-crawled baseline. **Verification step "particularly effective when we generalize the approach to languages beyond English."** This is a direct upgrade path for our 200K hand-rolled-noise pool.

**[Google production on-device LM training blog](https://research.google/blog/advances-in-production-on-device-language-models-with-differential-privacy/):** 30+ on-device Gboard LMs, 7+ languages, Portuguese/LatAm Spanish has 12,000+ devices per training round, 2000 rounds in 14 days. They pretrain on **multilingual C4** before language-specific tuning.

**[Apple ML Research](https://machinelearning.apple.com/research/learning-with-privacy-at-scale):** all federated learning + local DP. Less applicable to our offline-training paradigm. Moving toward 3B-param on-device foundation model with 2-bit QAT (long-term direction).

**[Gboard original FL paper, arxiv 1811.03604](https://arxiv.org/abs/1811.03604):** 1.4M-param CIFG (RNN), federated. **[DP follow-up, arxiv 2305.18465](https://arxiv.org/abs/2305.18465).**

### Round 2 — Fine-tuning theory + GEC

**Most important: [Does Prompt Loss Matter? arxiv 2401.13586](https://arxiv.org/abs/2401.13586)** and [Towards Data Science writeup](https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4/):

For short completions (~5 tokens, exactly our XBU correction span):
- Optimal PLW is in **[0, 0.1]**. PLW=0 and PLW≤0.1 perform nearly identically.
- **PLW=1 (no mask) is significantly worse**: 75% vs 80% accuracy on RACE benchmark.
- The "most impactful finding": monitor completion-only loss separately, even with PLW=1.

Translation: our 4a (PLW=0) is correct. Our 4b/4c (PLW=1, full sequence loss) is in the worst zone. **The fix is partial loss masking (PLW≈0.05) on clean tokens, not lowering typo_rate.**

[Sebastian Raschka's instruction-masking insights](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction): for limited data + short responses, masking helps. For larger data + long responses, full-sequence loss can help via regularization. Recommends empirical testing rather than dogma.

**Catastrophic forgetting:** documented across 1B-7B models. PEFT/LoRA does NOT prevent it. [Revisiting CF in LLM Tuning (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.249/) shows flat loss landscape correlates with less CF; **Sharpness-Aware Minimization (SAM)** is a clean mitigation.

**GEC literature:** decoder-only causal LMs are now competitive with BART/T5 for grammatical error correction. xfspell, NeuSpell are public spell-correction toolkits using seq2seq, with 7M+ / 1.6M training pair scale respectively (vs our 200K).

### Round 3 — Practical / community

- **FUTO themselves removed on-device LoRA fine-tune** ("never stabilized, battery drain, broken models"). The `finetune.cpp` in their tree IS that removed code — irrelevant to our offline pipeline.
- **No public FUTO fork has trained a non-English LM.** We're the first/only attempt.
- **[TinyStories, arxiv 2305.07759](https://arxiv.org/abs/2305.07759):** <10M-param LMs can be coherent if data is constrained. Validates 36M is plenty of capacity.
- **llama.cpp built-in `finetune` is for tiny models on CPU** — confirms our HF-based pipeline.
- **MobileLLM finding (from [On-Device LM Survey, arxiv 2409.00088](https://arxiv.org/html/2409.00088v1)):** for sub-1B params, **deep+thin > wide+shallow** by 2.7-4.3pp accuracy.

### Round 4 — Brazilian Portuguese particularities

**[TeenyTinyLlama-160m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m)** — game-changer:
- Apache 2.0 pt-BR Llama 2: 12 layers, 768 hidden, 12 heads, 32K SentencePiece vocab
- Trained on **6.2B pt-BR tokens** (Pt-Corpus + instruction data)
- Authors explicitly note model is "under-trained, can improve if further trained"
- Single A100, $500 budget, 36 hours
- Drop-in pretrain replacement, or distillation source

**Other pt-BR models** ([arxiv 2401.16640](https://arxiv.org/html/2401.16640v2)): Sabiá 7B/65B (too big), Albertina 100M-1.5B (encoder-only).

**[Penteado 2023 pt-BR GEC dataset, arxiv 2306.15788](https://arxiv.org/abs/2306.15788):** 4 categories — Grammar, Spelling, Internet, **Fast typing**. "Fast typing" is exactly our use case. Should replace/augment our 50-pair eval. Access via paper's supplementary material.

**NILC at USP** has a 78%-accuracy phonetic speller for pt-BR (heuristic, not LM) — sets a non-trivial baseline floor.

**ABNT2 layout details ([kbdlayout.info/KBDBR](http://www.kbdlayout.info/KBDBR/)):** ç has dedicated key right of L. Dead keys for diacritics. Our adjacency map needs `l↔ç` neighbor link.

**Brazilian texting style:** consonant-shortening (vc, tb, blz, hj) is language-specific. ~100% of Brazilians use WhatsApp. Our real-typo log heavily reflects this.

---

## 3. Cross-cutting synthesis

Three convergent observations across rounds:

**(A) The XBU format is unproven.** The FUTO wiki defines it but the wiki is incomplete. SwiftKey's production transformer skips autocorrect formats entirely. Our 12% top-5 at stage_a is the only evidence the XBU approach works for pt-BR — and that's a noisy datapoint. We should ablate.

**(B) The 4b loss formulation is mechanically wrong.** The Prompt Loss Weight literature directly predicts our outcome: PLW=1 (no masking) on short-completion data gives 75% vs PLW≤0.1 giving 80%. We landed at the worst end of the curve. Fix: PLW=0.05 in 4b/4c.

**(C) Data quality > quantity, and the upgrade is cheap.** Google's two-stage Gemini error-injection beat web-crawled data by 22.8% NWP. Our 200K hand-rolled-noise pool is exactly the artifact this method replaces. Regenerating with Claude API for pt-BR costs ~$30 and should give a 10-20% absolute top-5 jump on its own.

**Calibrated expectation:** SwiftKey's published transformer barely beats their GRU. The FUTO reference English model's 74% top-1 may be measured on a favorable test set. **A "good" pt-BR result is probably 30-40% top-5 on real typos, not 74%.** That's still a massive improvement over the broken v6 and worth shipping.

---

## 4. Revised v3 plan

**Ordered by impact ÷ cost.** Each phase has a stop-go decision before the next.

### Phase 0 — Cheap code fixes (1 day, no GPU)

User decision: **skip v7 packaging**, go straight to v8. Accepted risk: no on-device eval sanity check until v8 ships. Mitigation: bring the smoke-test forward to Phase 7 (pre-ship), and if on-device diverges, do an emergency patch run.

0.1. **Add PLW=0.05 to `04b_finetune_fulltext.py`** — single highest-leverage code change. Full weight on XBU spans, 0.05 weight on clean tokens. Predicted to fix the stage_b collapse.

0.2. **Add real-typo eval callback every 500 steps** to all training scripts. CSV trajectory + curve plot. Hard prerequisite for all v3 runs.

0.3. **Add ç to ABNT2 adjacency map** in `lib_typo_synthesis.py`.

0.4. **Add SAM optimizer** for 4b/4c (catastrophic forgetting mitigation).

### Phase 1 — Validate fixes via small ablations (~30min wall-clock, parallel 2-GPU)

Each run starts from `pretrain_big/base`, 5K steps at half scale, evals against hold-out:
- A1: 4a at PLW=0 — confirm 12% top-5 baseline
- A2: 4a at PLW=0.05 — test non-zero PLW
- B1: 4b at PLW=0.05, typo_rate=0.33 (wiki value) — predicted: stops collapsing
- B2: B1 + SAM — predicted: best small-scale result

**Parallel GPU strategy:**
- **Unraid 3090 (24GB)**: A1 + A2 sequentially (4a is light; ~5GB VRAM each)
- **Gaming desktop 5070 Ti (16GB)**: B1 + B2 sequentially (4b at half scale; reduce micro_batch from 24→12 to fit 16GB)
- Two machines run in parallel → wall time ~30min instead of ~2h serial

Decision: B1 should show top-5 ≥ 12% by step 5K. If not, recipe is broken at a deeper level than loss masking.

### Phase 2 — Right-sized synthetic data via Claude API (1 day, <$5)

**Right-sizing**: stage_a's 12% top-5 came largely from the 343 real pairs (25% mix), not the 200K hand-rolled synth pool. The 200K pool was over-engineered. New target: **~750 high-quality Claude-generated pairs** (~2.2× the real pool — "just barely more than double"). At Sonnet 4.6 prices: roughly 150K tokens total, ~$1-2.

Seed strategy informed by [Gimenes 2015 (Computational Linguistics, MIT Press)](https://aclanthology.org/J15-1011.pdf):
- **54.9% of all pt-BR spelling errors are diacritic misuse; 51.5% are diacritic OMISSION.** Confirms our category split: accent_only should be the largest synth category.
- **Cedilla (ç) gets its own category** because ABNT2 has a dedicated key while US-Accents requires a 2-key composite. Our user is on ABNT2, so ç-omission is a primary error class.
- Frequency-based confusion patterns observed across multiple sources:
  - **x ↔ ch** (mexer/mecher)
  - **j ↔ g** (especially before e/i)
  - **s ↔ z, ç ↔ ss, s ↔ ç**
  - **e ↔ i in unstressed syllables** (privilégio → previlégio, simplesmente → simplismente)
  - **dropped r** (apropriado → apropiado)
  - **a gente vs agente, por que vs porque, mas vs mais, à vs a** (semantic-near-homophones)
- Most-misspelled list ([Pensar Cursos](https://www.pensarcursos.com.br/blog/palavras-que-os-brasileiros-mais-escrevem-errado/), [Jusbrasil 100 erros](https://www.jusbrasil.com.br/artigos/os-100-erros-mais-comuns-de-lingua-portuguesa/635153582), [Toda Matéria](https://www.todamateria.com.br/erros-de-portugues/), [Dicio](https://www.dicio.com.br/erros-de-ortografia/)): exceção (esceção/excessão), beneficente, subsídio, privilégio (previlégio), empecilho (impecilho), apropriado (apropiado), simplesmente (simplismente), mexer (mecher).
- WhatsApp study findings ([Correio Braziliense](https://www.correiobraziliense.com.br/cbradar/os-erros-de-portugues-mais-comuns-em-mensagens-de-whatsapp-que-quase-todo-mundo-ainda-comete/)): mobile screens cause **wrong-key adjacency** (already in our model) + **z/s and ç/ss confusion**. Multi-error phrases dominate vs single-error.

Implementation:
- Build `scripts/08_claude_synth.py`. Seed clean sentences from BrWaC / Carolina.
- Use Google's two-stage prompt template, parameterized with our category priors (54% accent_only, etc.).
- Inject the most-misspelled-word list as one of the few-shot examples.
- Round-trip verify: re-correct, keep only matches.
- Target ~750 pairs. Manual eyeball sample of 50 for plausibility before adopting.
- Replace `notes/synth_typos.json` with the new pool. Re-run B2 from Phase 1.

Expected: 5-10pp absolute top-5 improvement on top of Phase 1 gains (less than the original 200K plan because the data quantity matters less than the loss-mask fix).

### Phase 3 — XBU ablation (~half day wall-clock, parallel 2-GPU)

Train two checkpoints from the validated Phase 1 config:
- **3090 (Unraid)**: XBU-format twin (current paradigm)
- **5070 Ti (gaming)**: plain NWP twin (no XBU/XBC/XEC)

Both run concurrently. Smoke-test both on phone. If NWP-twin produces good autocorrect on-device, drop the format entirely (major complexity reduction). If only XBU-twin works, confirm the format is load-bearing and document why.

### Phase 4 — TeenyTinyLlama-160m direct (conditional, 1 day, 3090 only)

User decision: **if 36M can't hit the ship floor → use TTL-160m direct** (no distillation). 4× larger model on phone is acceptable. Apache-2.0 pt-BR pretrained on 6.2B tokens.

Trigger: best 36M Phase 1-3 result <30% top-5. Run the validated Phase 1-3 recipe on top of TTL-160m as the new base. Re-evaluate against ship floor.

**GPU note**: 160M model + seq_len 512 + reasonable batch → ~14-18GB VRAM. **Use 3090 (24GB)**; the 5070 Ti at 16GB is too tight. The 5070 Ti can still run smaller experiments in parallel (e.g. small-scale typo_rate sweeps).

### Phase 5 — Richer eval baseline (parallel, no GPU)

Acquire [Penteado 2023 pt-BR GEC dataset](https://arxiv.org/abs/2306.15788) via paper supplementary. Run all v3 checkpoints against the "Fast typing" + "Internet" categories. Establishes publicly-comparable numbers.

### Phase 6 — Final big run (~6h on 3090, 5070 Ti runs eval / Penteado bench in parallel)

After Phase 1-5 validation, ONE full-scale 4b run with the validated config on the 3090. Not before.

**Parallel work for the 5070 Ti during this 6h**: run continuous evaluation of intermediate checkpoints (every 2500 steps the 3090 saves) against the Penteado pt-BR GEC dataset categories. The eval rig becomes a passive consumer of checkpoints, producing a richer evaluation matrix while training proceeds.

### Phase 7 — Package v8 + on-device smoke test (1h)

GGUF v2, Q6_K output, F16 elsewhere, FUTO feature flags. ADB push. **Pre-ship**: side-load, type the 50 hold-out typos manually, compare on-device top-5 to Python eval. If divergence >5pp, hold release and triage harness. This is the deferred sanity check from "no v7" decision.

**Ship gate**: top-5 ≥ 30% on real hold-out OR top-5 ≥ 40% on accent_only/cedilla_only categories (the dominant pt-BR error classes per Gimenes 2015). If gate failed, retry with TTL-160m base (Phase 4).

---

## 5. Decisions (locked in 2026-05-12)

1. **No v7 ship.** Wait for v8. Accepted risk: no early on-device sanity check; mitigated by bringing the smoke test to Phase 7 pre-ship.
2. **XBU format: Phase 3 ablation first.** Run plain NWP twin against XBU twin in parallel on both GPUs.
3. **Claude API budget: dropped as a question.** At 750-pair scale the cost is <$5, not decision-blocking.
4. **Fallback model: TTL-160m direct** (no distillation). 160M on phone is acceptable.
5. **Ship floor: 30-40% top-5 overall on real typos**, alternate gate is beat-NILC-on-accent_only (78% phonetic-speller baseline).
6. **Use both GPUs in parallel where possible.** 3090 (24GB, Unraid) for heavier runs and TTL-160m; 5070 Ti (16GB, gaming desktop) for half-scale ablations, parallel twins, and continuous eval during long runs.

---

## 6. Things we're explicitly NOT doing

- Reproducing the English reference checkpoint — deferred until after v8 ships
- Federated / on-device fine-tune — FUTO themselves gave up here
- Full lib_typo_synthesis calibration — low impact vs PLW fix
- 72M-param scale-up — only if Phase 4 (TTL pivot) still fails

---

## 7. Source index

### FUTO sources (verified)
- [docs.keyboard.futo.org](https://docs.keyboard.futo.org/) — official docs
- [github.com/futo-org/android-keyboard](https://github.com/futo-org/android-keyboard) — mirror of GitLab source
- [gitlab.futo.org/keyboard/keyboard-wiki Keyboard-LM-docs](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs) — official LM wiki (incomplete)
- [gitlab.futo.org lm-2-finetuning branch](https://gitlab.futo.org/keyboard/latinime/-/tree/lm-2-finetuning) — removed on-device LoRA code

### Big-keyboard papers
- [SwiftKey transformer, arxiv 2505.05648](https://arxiv.org/html/2505.05648)
- [Google Synthetic+Federated blog](https://research.google/blog/synthetic-and-federated-privacy-preserving-domain-adaptation-with-llms-for-mobile-applications/)
- [Google on-device LM training blog](https://research.google/blog/advances-in-production-on-device-language-models-with-differential-privacy/)
- [Gboard FL original, arxiv 1811.03604](https://arxiv.org/abs/1811.03604)
- [Gboard DP, arxiv 2305.18465](https://arxiv.org/abs/2305.18465) + [ACL Industry 2023](https://aclanthology.org/2023.acl-industry.60/)
- [Apple Learning with Privacy at Scale](https://machinelearning.apple.com/research/learning-with-privacy-at-scale)

### Fine-tuning theory
- [Does Prompt Loss Matter, arxiv 2401.13586](https://arxiv.org/abs/2401.13586) + [TDS writeup](https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4/)
- [Raschka instruction masking](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction)
- [Catastrophic Forgetting in LLM Tuning, EMNLP 2024 findings.249](https://aclanthology.org/2024.findings-emnlp.249/)

### pt-BR resources
- [TeenyTinyLlama-160m HF](https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m) + [paper arxiv 2401.16640](https://arxiv.org/html/2401.16640v2)
- [Sabiá-2, arxiv 2403.09887](https://arxiv.org/abs/2403.09887)
- [Penteado pt-BR GEC, arxiv 2306.15788](https://arxiv.org/abs/2306.15788)
- [Portuguese-NLP resource list](https://github.com/ajdavidl/Portuguese-NLP)
- [ABNT2 layout details](http://www.kbdlayout.info/KBDBR/)

### Tools / hobbyist
- [xfspell](https://github.com/mhagiwara/xfspell)
- [NeuSpell](https://github.com/neuspell/neuspell)
- [TinyStories, arxiv 2305.07759](https://arxiv.org/abs/2305.07759)
- [On-Device LM Survey, arxiv 2409.00088](https://arxiv.org/html/2409.00088v1)
