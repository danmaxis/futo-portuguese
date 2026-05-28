<p align="right"><a href="README.md">🇺🇸 English</a> · <strong>🇧🇷 Português</strong></p>

# FUTO Keyboard — modelo de linguagem em Português brasileiro

Um transformer de 36M parâmetros, arquitetura Llama, que roda dentro do [FUTO Keyboard](https://keyboard.futo.org) e te dá autocorreção e previsão da próxima palavra **direto no celular**, sem mandar nada pra nuvem, em **Português brasileiro**.

O FUTO só publica um modelo em inglês. A posição oficial deles é "estamos trabalhando nos outros", sem prazo, e eles aceitam explicitamente o carregamento de modelos de terceiros no mesmo formato — só que a spec do formato no wiki público está incompleta e várias etapas de integração não são óbvias. Esse repo fecha essas lacunas **e** publica o modelo pt-BR treinado.

> **Última release:** [`v8.2`](../../releases/latest) — GGUF Q6_K, 62 MB, carrega no FUTO e roda num Android de verdade.

---

## TL;DR

| | |
|---|---|
| Arquitetura | Llama, 8 camadas, 512 hidden, 36M parâmetros (idêntico à referência em inglês do FUTO) |
| Tokenizer | SentencePiece BPE, vocab de 15008, com 300 slots `user_defined` fixos |
| Corpus de pretrain | ~4B tokens pt-BR (BrWaC, OSCAR, Wikipedia, Carolina, OpenSubtitles, CORAA) |
| Hardware pra reproduzir | 1× RTX 3090 (24 GiB) ou qualquer GPU de consumidor com 16+ GiB |
| Tempo de parede pra reproduzir | ~30–50h pretrain + ~3,5h fine-tune + <30min empacotamento |
| Eval em typos reais (50 pares de holdout) | **60% top-1 / 72% top-5** (v8.2) |
| Melhor categoria | **accent_only: 88% top-5** (a classe de erro dominante no pt-BR) |
| Pontos fracos (ainda) | hybrid (multi-erro), prefix completion |

Toda a pipeline de treino, a spec de formato que engenheiramos reverso, e o GGUF treinado estão sob licença MIT. **Você deveria conseguir fazer isso para o seu idioma em um fim de semana, numa GPU única de consumidor.**

---

## Por que isso existe

Um teclado em que você digita o dia inteiro é o melhor lugar único pra um LM privado pequeno. O FUTO Keyboard já roda um, on-device, via [llama.cpp](https://github.com/ggerganov/llama.cpp) — mas só em inglês, e o wiki oficial sobre o formato é incompleto o bastante pra obrigar a engenharia reversa de várias coisas a partir do GGUF binário e de um *crash backtrace*.

Esse repo é o registro de fazer isso para o Português brasileiro do início ao fim. E também é um **caminho das pedras** pra quem quiser fazer pro *seu* idioma: cada armadilha que pisamos, cada beco sem saída descartado, cada número de loss, em ordem versão-por-versão.

---

## A jornada

Cada versão mira um número real, num holdout real (digitado no celular de verdade e depois corrigido pelo mesmo humano). As categorias são autoexplicativas: `accent_only` (cafe → café), `adjacency_or_short_edit` (um deslize de tecla), `prefix_completion` (digitou as 2–3 primeiras letras, espera o resto), `hybrid` (multi-erro), `cedilla_only` (c → ç).

### v2 — o muro (~6h, maio de 2026)

Construímos a pipeline inteira de ponta a ponta. Seguimos o wiki público do FUTO conforme escrito.

| Stage | top-1 | top-5 |
|---|---|---|
| stage_a | 1/50 (2%) | 6/50 (12%) |
| stage_b | **0/50** | **0/50** |
| stage_c | **0/50** | **0/50** |

Mode collapse: o modelo aprendeu a sempre emitir o texto corrigido literal com que estava sendo treinado, ignorando o *prompt* de teclas. Cinco hiperparâmetros tinham sido mudados de uma vez — nenhum podia ser responsabilizado individualmente.

**Lição:** *não mude cinco coisas de uma vez; não confie numa receita incompleta.*

### v3 — o diagnóstico (pesquisa + ablações, ~1 dia, 12/05/2026)

Quatro rodadas de pesquisa bibliográfica ([artigos da SwiftKey/Gboard/Apple](notes/v3_research_synthesis.md), teoria de fine-tuning, *prompt-loss masking*, particularidades do pt-BR) mais cinco ablações pequenas de **uma variável só** divididas entre 5070 Ti e 3090. Duas causas reais apareceram:

1. **A formulação da loss em 04b/04c estava mecanicamente errada.** Calculava loss sobre a *sequência inteira* (PLW=1.0). ~90% do gradiente vinha de tokens limpos em pt-BR; os 10% de tokens de autocorreção XBU eram afogados. Correção: PLW=0.05 (conforme [arxiv:2401.13586](https://arxiv.org/abs/2401.13586)), que escala a loss dos tokens limpos pra baixo sem zerar. E também: um bug *off-by-one* deslocando os labels um token cedo.
2. **SAM (Sharpness-Aware Minimization) atrapalhou o stage_a.** Testamos por causa de *catastrophic forgetting*; ablamos, regrediu em todos os checkpoints contra AdamW puro, descartado.

| Ablação | top-1 | top-5 | Veredito |
|---|---|---|---|
| A1 (PLW=0.0) | 3/50 (6%) | 6/50 (12%) pico, faz overfitting | colapso baseline |
| A2 (PLW=0.05) | 4/50 (8%) | 8/50 (16%) | **correção confirmada** |
| A5 (PLW=0.05 + SAM) | 0 | 0–1 | SAM descartado |
| **B1** (4b, PLW=0.05) | 8/50 (16%) | **15/50 (30%)** no passo 4000 | **gate de release alcançado** |

Critério de stop-go (≥30% top-5) aprovado. O pool de 200K pares sintéticos feitos à mão estava dimensionado certo; a geração via Claude-API que tínhamos planejado ficou pra depois. **Hora de soltar um v8 de verdade.**

### v8 — o primeiro ship (~4h na 3090, 12/05/2026)

Phase 4 completa (a + b + c) com PLW=0.05, correção do off-by-one, sem SAM. Mix de typos reais 25%, 200K sintéticos + 343 reais. Empacotado como GGUF v2 com `output.weight` Q6_K + resto F16 (idêntico à referência em inglês).

| Categoria | n | top-1 | top-5 |
|---|---|---|---|
| **Geral** | 50 | **36,0%** | **56,0%** |
| accent_only | 17 | 52,9% | **88,2%** |
| adjacency_or_short_edit | 21 | 42,9% | 61,9% |
| hybrid | 5 | 0% | 0% |
| prefix_completion | 7 | 0% | 0% |

Excluindo as categorias categoricamente difíceis (hybrid + prefix), o v8 bateu **74% top-5 nos pares "solucionáveis"**. O `v8.gguf` carrega no FUTO Keyboard 0.1.27 no Android e prevê. Soltado como o primeiro artefato real.

**Lição:** *corrige o bug mecânico, e o resto da receita estava basicamente correto.*

### v8.1 — a tentativa de afiar (~5h, 22/05/2026) — NÃO foi shipado

Duas mudanças mirando o top-1 e as categorias fracas:

- **Pool de typos reais atualizado** de 343 → 393 pares únicos após dedup; carvamos um **holdout v8.1 disjunto de 58 pares**, intencionalmente sobre-representando prefix e hybrid.
- **Pares sintéticos via Claude CLI** (`claude -p` no shell, sem chave de API) — 500 gerais + 250 mirados em pontos fracos, gerados com o padrão de prompt em dois estágios do [Google Gemini](https://research.google/blog/improving-mobile-keyboard-language-models/).
- **PLW_C = 0,02** no stage_c pra afiar top-1.

Resultado no holdout v8: **44% top-1 / 64% top-5** (+8pp / +8pp sobre v8). No holdout v8.1 mais difícil: 39,7% top-1 / 55,2% top-5.

Só que **NWP regrediu**: o top-3 conversacional de próxima palavra caiu de 6,1% (v8) pra 0,7% (v8.1, mascarado). A receita mais afiada matou o contexto livre. **Não foi shipado.**

**Lição:** *não otimize uma métrica sem olhar as outras.* Salvo como memória de `feedback` pra não repetirmos.

### v8.2 — a receita melhor (~3,5h de fine-tune, 22–23/05/2026) — **release atual**

Recuamos do PLW_C=0,02. Mantemos o pool Claude de pontos fracos. Adicionamos um **continue-pretrain num corpus conversacional** (~42K snippets estilo mensagem) antes do fine-tune, pra dar ao modelo base um prior mais forte de registro casual.

Tempo de parede da Phase 4 numa 3090 sozinha:

| Stage | Steps | Tempo |
|---|---|---|
| 4a (autocorreção isolada) | 3000 | **7m 04s** |
| 4b (autocorreção em contexto) | 12000 | **2h 40m 51s** |
| 4c (adaptação conversacional) | 10000 | **48m 56s** |

Resultado no mesmo holdout v8 de 50 pares:

| Categoria | n | top-1 | top-5 |
|---|---|---|---|
| **Geral** | 50 | **60,0%** | **72,0%** |
| accent_only | 17 | 82,4% | 88,2% |
| adjacency_or_short_edit | 21 | 66,7% | 81,0% |
| hybrid | 5 | 20% | 40% |
| prefix_completion | 7 | 14,3% | 28,6% |

No holdout v8.1 mais difícil: 39,7% top-1 / **63,8% top-5** (vs 55,2% do v8.1) — +8,6pp de top-5 com o mesmo top-1, num holdout que intencionalmente sobre-representa as categorias difíceis. NWP top-3 mascarado é 4,7% (ainda abaixo de 6,1% do v8 — regressão honesta, parte do sinal conversacional livre foi sacrificada pra ganhar precisão em typos; estamos monitorando).

Esse é o **release destacado**. O GGUF v8 continua disponível pra quem quiser o artefato original já comprovado.

**Lição:** *meça em múltiplos holdouts e em NWP de texto livre, não só num único conjunto de typos.*

---

## Visão geral — comparativo de versões

| Versão | Top-1 (holdout v8) | Top-5 (holdout v8) | NWP top-3 (mascarado) | Foi shipada? |
|---|---|---|---|---|
| v2 stage_a | 2% | 12% | — | não (colapsou) |
| v2 stage_b/c | 0% | 0% | — | não (colapsou) |
| **v8** | 36% | 56% | **6,13%** | **sim (legado)** |
| v8.1 | 44% | 64% | 0,75% | não (regressão de NWP) |
| **v8.2** | **60%** | **72%** | 4,66% | **sim (atual)** |

---

## Reproduzir pra sua língua

Referência técnica completa passo-a-passo: [**GUIDE.md**](GUIDE.md) (Inglês) · [**GUIDE.pt-BR.md**](GUIDE.pt-BR.md) (Português). O que realmente precisa:

1. **Phase 0** — extrair o GGUF de referência em inglês do FUTO como sua spec (`scripts/00_inspect_reference.py`). O formato é amplamente não-documentado; o binário é a fonte da verdade.
2. **Phase 1** — montar um corpus de 3–5B tokens no idioma alvo (`scripts/01_build_corpus.py`, faz streaming do HuggingFace). Sinal mais forte de registro casual que achamos: OpenSubtitles.
3. **Phase 2** — treinar um tokenizer SentencePiece BPE, **vocab 15008**, com os 300 símbolos user-defined fixados nos IDs 4..303 na ordem *exata* do FUTO (`scripts/02_train_tokenizer.py`).
4. **Phase 3** — pretrain (`scripts/03_pretrain.py`, ~100K passos, 30–50h na 3090).
5. **Phase 4** — fine-tune de autocorreção em três stages (`scripts/04a_*`, `04b_*`, `04c_*`). **Use PLW=0,05** nos stages b e c — essa é a única coisa mecânica que o wiki erra por omissão.
6. **Phase 5+6** — converter pra GGUF, fazer downgrade pra GGUF v2, patch dos metadados FUTO, **quantizar `output.weight` em Q6_K** (output em F16 trava a JNI do FUTO no segundo toque). Scripts: `05_to_futo_gguf.py`, `06_patch_metadata.py`, `06b_downgrade_v2.py`.
7. **Smoke test num Android de verdade antes de comemorar.** Eval em Python ≠ eval no dispositivo (quant diferente, feature flags diferentes, sampler diferente).

Orçamento: **um fim de semana numa GPU de consumidor de 24 GiB**, mais algumas horas de familiaridade com Python. A coisa mais difícil sozinha é o layout dos slots do tokenizer — é por isso que `notes/reference_slot_map.md` existe.

---

## As cinco armadilhas, em uma linha cada

(Discussão completa: [GUIDE.md §12](GUIDE.md#12-the-five-gotchas-in-one-place).)

1. **O formato do prompt é uma sequência de teclas (tokens `<CHAR_X>`), NÃO texto literal.** Essa é a alegação errada de mais peso no wiki público.
2. **`<CHAR_A>..<CHAR_Z>` precisam ser 26 IDs de token contíguos.** O C++ do FUTO faz aritmética de ponteiros sobre eles.
3. **GGUF tem que ser v2.** O llama.cpp escreve v3 por padrão; o llama.cpp embarcado no FUTO não lê.
4. **`output.weight` precisa ser Q6_K.** Output em F16 → `SIGSEGV` no segundo toque. Não está no wiki.
5. **PLW tem que ser ~0,05 na Phase 4b/c.** Caso contrário o modelo entra em mode collapse e sempre emite o texto corrigido. Não está no wiki.

---

## Limitações honestas

- Transformers de teclado em produção (os números publicados pela SwiftKey) ganham ~**1 ponto percentual** em NWP sobre um baseline GRU. Não espere milagre. A vitória é em cobertura de categoria de autocorreção, não em top-1 puro da próxima palavra.
- Precisão em typos reais depende do *seu* padrão de digitação — um holdout de 50 pares é informativo, não definitivo. Construa o seu próprio holdout a partir do seu próprio log de typos.
- Versões do FUTO Keyboard **posteriores à 0.1.27** têm uma regressão upstream nas sugestões de NWP com prefixo vazio — bug do app, não do modelo. Diagnóstico e workaround: [`notes/futo_regression.md`](notes/futo_regression.md).
- `prefix_completion` e `hybrid` (multi-erro) continuam as categorias mais difíceis. O v8.2 só ataca parcialmente. Problema em aberto.

---

## Instalar o modelo

1. Baixa o `futo_pt_br_v8_2.gguf` da [release mais recente](../../releases/latest).
2. Carrega pelo FUTO Keyboard em settings → Models → Import. Instruções completas e walkthrough do ADB: [GUIDE.md §11](GUIDE.md#11-side-loading-and-reproducing-a-crash).
3. Se travar, manda o `adb logcat` que a gente olha.

---

## Agradecimentos

- **FUTO** por shipar um teclado privado on-device e por explicitamente apoiar modelos de terceiros.
- **llama.cpp** por ser o runtime que torna esse tamanho de modelo viável num celular.
- **HuggingFace** e os mantenedores de BrWaC, OSCAR, Wikipedia-pt, Carolina, OpenSubtitles, Common Voice e CORAA.
- **Artigos**: arxiv:2401.13586 (Prompt Loss Weight), arxiv:2505.05648 (Transformers privacy-preserving da SwiftKey), o post do Google sobre LMs de teclado mobile (geração sintética de typos em dois estágios).
- O wiki do FUTO Keyboard, incluindo as partes que estão erradas — errar essas partes é o que produziu esse guia.

## Licença

- Todos os scripts Python, scripts shell e prosa nesse repo (esse README, `GUIDE.md`, `GUIDE.pt-BR.md`, arquivos em `notes/`): **MIT**. Veja [`LICENSE`](LICENSE).
- Os pesos do modelo `.gguf` são liberados sob a **mesma licença MIT** — sinta-se à vontade pra usar, redistribuir, modificar.
- Os dados de treino **não** são redistribuídos; só referências aos datasets públicos e os scripts que baixam.
- O FUTO Keyboard em si está sob a FUTO Source First License 1.1 (projeto separado, não embutido aqui).
