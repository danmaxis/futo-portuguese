# Treinando um transformer do FUTO Keyboard pra sua língua — um guia prático

Esse documento é pra quem quer treinar e side-loadar um transformer customizado dentro do [FUTO Keyboard](https://keyboard.futo.org) e ter **autocorreção + previsão da próxima palavra, on-device e privado**, num idioma pro qual o FUTO não publica modelo.

O FUTO só publica um transformer Llama de ~36M parâmetros, em inglês. A posição oficial deles é "estamos trabalhando nos outros idiomas", sem prazo firme. Eles aceitam explicitamente carregar modelos de terceiros no mesmo formato, mas a spec do formato no wiki público é incompleta e a integração tem várias armadilhas que não são óbvias. Esse guia fecha essas lacunas.

Foi escrito a partir de uma rodada real, de ponta a ponta: treinar um modelo em Português brasileiro, bater em cada muro, e shipar um `.gguf` side-loadável que carrega e roda num Android de verdade. Onde o wiki público do FUTO está certo, a gente cita. Onde está errado ou incompleto, a gente diz e dá a resposta verificada.

Esse guia assume que você consegue:

- ler Python e C++ num nível suficiente pra acompanhar scripts de treino e um crash backtrace,
- operar uma máquina Linux com GPU NVIDIA,
- usar SSH, Docker e Android Debug Bridge (`adb`).

Se você é novo em treino de ML em si, esse guia sozinho não vai bastar — ele foca nas partes específicas do FUTO.

---

## Sumário

1. [O que é, de fato, o transformer do FUTO](#1-what-futos-transformer-actually-is)
2. [Hardware e orçamento de tempo](#2-hardware-and-time-budget)
3. [A spec não-documentada: como seu modelo precisa ser](#3-the-undocumented-spec-what-your-model-has-to-look-like)
   1. [Arquitetura (a parte fácil)](#31-architecture-the-easy-part)
   2. [O tokenizer: 300 símbolos user-defined em slots fixos](#32-the-tokenizer-300-user-defined-symbols-at-fixed-slots)
   3. [O formato de prompt por keypress (NÃO é texto literal)](#33-the-keypress-prompt-format-not-literal-text)
   4. [Campos de metadados GGUF](#34-gguf-metadata-fields)
4. [Visão geral da pipeline](#4-pipeline-overview)
5. [Phase 0 — extrair o modelo oficial como sua spec](#5-phase-0--extract-the-official-model-as-your-spec)
6. [Phase 1 — montar um corpus no idioma alvo](#6-phase-1--assemble-a-target-language-corpus)
7. [Phase 2 — treinar o tokenizer SentencePiece](#7-phase-2--train-the-sentencepiece-tokenizer)
8. [Phase 3 — pretrain do modelo base](#8-phase-3--pretrain-the-base-model)
9. [Phase 4 — fine-tune de autocorreção (3 stages)](#9-phase-4--autocorrect-fine-tune-3-stages)
10. [Phase 5 — empacotar como GGUF compatível com FUTO](#10-phase-5--package-as-a-futo-compatible-gguf)
11. [Side-loading e reprodução de um crash](#11-side-loading-and-reproducing-a-crash)
12. [As cinco armadilhas, num lugar só](#12-the-five-gotchas-in-one-place)
13. [Metodologia de avaliação](#13-evaluation-methodology)
14. [O que esse guia não cobre](#14-what-this-guide-does-not-cover)
15. [Agradecimentos e licenciamento](#15-acknowledgements-and-licensing)

---

<a name="1-what-futos-transformer-actually-is"></a>
## 1. O que é, de fato, o transformer do FUTO

O FUTO Keyboard roda dois algoritmos de previsão em paralelo e funde os dois:

1. Um engine clássico estilo AOSP, baseado em dicionário e bigrama. Sua língua provavelmente já tem um arquivo de dicionário shipado (`dictionaries/<lang>_wordlist.combined.gz` em [futo-org/android-keyboard](https://github.com/futo-org/android-keyboard)).
2. Um **transformer de arquitetura Llama** treinado pra autocorreção e previsão da próxima palavra. O transformer tem ~36M parâmetros, roda on-device via [llama.cpp](https://github.com/ggerganov/llama.cpp), usa um tokenizer SentencePiece embarcado, e hoje só vem em inglês.

Se você habilitar só o (1), tem previsões em nível de spellcheck: completar palavras, autocorreção simples, bigrama aprendido. Adequado pra muitos idiomas. Se você também habilitar o (2) com um modelo bem treinado na sua língua, ganha autocorreção sensível ao contexto e previsão gramatical da próxima palavra.

Esse guia é sobre produzir o (2).

O verbete do wiki que aponta pro formato está em <https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs>. Leia uma vez. Aí volta aqui, porque várias afirmações lá são enganosas e uma parte crítica está completamente faltando.

---

<a name="2-hardware-and-time-budget"></a>
## 2. Hardware e orçamento de tempo

**Mínimo pra shipar um modelo funcional**: uma GPU NVIDIA de consumidor com **16+ GiB de VRAM**. Uma GPU de 24 GiB é mais confortável. Treino em CPU é inviável nesse tamanho.

**Tempo de parede realista pra uma rodada de qualidade de verdade** em uma RTX 3090:

| Phase | Tempo |
|---|---|
| Montagem do corpus (3-5B tokens) | 1-3 horas, gargalo de rede |
| Treino do tokenizer | 15-60 minutos, gargalo de CPU + RAM |
| Pretrain (~100k passos de otimizador) | 30-50 horas |
| Fine-tune (Phase 4a + 4b + 4c) | 3-8 horas |
| Montagem do GGUF + side-load | <30 minutos |

Uma **rodada mini de validação** (uma única fonte de corpus, 20k passos de pretrain) leva cerca de 14 horas no total e basta pra verificar sua pipeline ponta a ponta antes de comprometer dias numa rodada de verdade.

**RAM**: o treino do tokenizer SentencePiece é gargalo de memória e não tem aceleração de GPU. Pra um corpus de vários GiB você quer **32+ GiB de RAM** no host que faz o treino do tokenizer. Um host de 16 GiB vai swapar pesado e ficar arrastando. Veja [Phase 2](#7-phase-2--train-the-sentencepiece-tokenizer).

**Disco**: 50-100 GiB livres pra shards do corpus + checkpoints. Artefatos pesados podem morar numa máquina separada acessada por SSH; só o GGUF final de ~62 MB precisa sair do rig de treino.

---

<a name="3-the-undocumented-spec-what-your-model-has-to-look-like"></a>
## 3. A spec não-documentada: como seu modelo precisa ser

Antes de escrever qualquer código: seu objetivo é um arquivo `.gguf` que o app Android do FUTO valida e carrega em runtime. A validação é estrita e parcialmente opaca. O caminho mais rápido pra acertar é **extrair o modelo de referência em inglês e reproduzir o layout dele**.

```bash
hf download breadlicker45/futo-keyboard-lm --local-dir reference_model/
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
       reference_model/ml4_1_f16_meta_fixed.gguf > reference_metadata.txt
```

`reference_metadata.txt` é a fonte da verdade. Toda seção abaixo cita ele.

<a name="31-architecture-the-easy-part"></a>
### 3.1 Arquitetura (a parte fácil)

Um config Llama padrão:

```python
LlamaConfig(
    vocab_size=15008,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,        # MHA, sem GQA
    max_position_embeddings=2048, # NÃO 512 — o wiki sugere 512 mas a referência é 2048
    rms_norm_eps=1e-6,            # NÃO 1e-5 — o wiki estava errado
    rope_theta=10000.0,
    tie_word_embeddings=False,
)
```

Isso é exatamente a referência. ~36M parâmetros no total. RoPE são os defaults padrão do llama.cpp. **Não mude esses valores** — o C++ em `LlamaAdapter` valida alguns deles.

<a name="32-the-tokenizer-300-user-defined-symbols-at-fixed-slots"></a>
### 3.2 O tokenizer: 300 símbolos user-defined em slots fixos

Essa é a parte que o wiki erra mais. O tokenizer tem **vocab de 15008**, dividido assim:

| Faixa de ID | Quantidade | Conteúdo |
|---|---|---|
| 0..3 | 4 | `<pad>`, `<s>`, `</s>`, `<unk>` (controle + unknown) |
| 4..303 | **300** | **Símbolos user-defined (o layout importa — veja abaixo)** |
| 304..559 | 256 | Byte-fallback `<0x00>`..`<0xFF>` |
| 560..15007 | 14448 | Peças BPE aprendidas do corpus (sua língua preenche isso) |

Os 300 símbolos user-defined são um layout estruturado, não 300 strings quaisquer. Lendo a referência e o source C++ do FUTO Android em `native/jni/src/ggml/LanguageModel.cpp` e `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp`, dá pra ver essas restrições:

```
Índices 4..27   — <FUTO0>..<FUTO23>  (24 slots reservados/inertes, filler)
Índices 28..173 — slots de conteúdo (contrações/palavras em inglês na referência)
                  → SUBSTITUIR pelos equivalentes do seu idioma
Índice 174      — <XBU>     ESTRUTURAL: autocorreção "begin user input"
Índice 175      — <XBC>     ESTRUTURAL: autocorreção "begin correction"
Índice 176      — <XEC>     ESTRUTURAL: autocorreção "end correction"
Índices 177..181— <XC0>..<XC4>  (só <XC0> é referenciado; o resto é reservado)
Índices 182..207— <CHAR_A>..<CHAR_Z>  ESTRUTURAL: tokens por keypress
                  ⚠️ ESSES 26 IDs PRECISAM SER CONTÍGUOS E SEQUENCIAIS.
                  O C++ em LanguageModel.cpp resolve <CHAR_A> pelo nome,
                  mas calcula o resto como LETTERS_TO_IDS[0] + i.
Índices 208..263— mais slots de conteúdo (SUBSTITUIR pelo seu idioma)
Índices 264..303— conjunto de emoji (40 slots; pode manter ou curar)
```

#### Comportamento de resolução no C++ (verificado, não apenas suposto)

A biblioteca Android usa dois mecanismos de lookup diferentes:

- **Por nome** (via `spm.PieceToId(...)` no SentencePiece embarcado): `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, `<CHAR_A>`, `▁` (marcador de espaço do SP). O ID deles pode estar em qualquer lugar, desde que o nome resolva pra um índice diferente de zero.
- **Por índice computado** (aritmética de ponteiros): `<CHAR_B>` até `<CHAR_Z>` são lidos como `LETTERS_TO_IDS[0] + i`. Então os 26 símbolos `<CHAR_*>` **precisam ocupar 26 IDs contíguos** no SentencePiece. O jeito mais simples de garantir isso é listar eles sequencialmente em `user_defined_symbols` — o SentencePiece preserva a ordem de declaração.

Se qualquer um de `<XBU>`, `<XBC>`, `<XEC>`, `<XC0>`, `<CHAR_A>` resolver pra 0 (= `<unk>`), o C++ dispara assert e crasha no load do modelo. Então todos precisam estar presentes no seu tokenizer.

#### Os slots filler `__FUTO0..23` e os emoji

Esses são inertes — nenhum código referencia eles por nome e nenhum lookup de embedding usa esses índices especificamente. Você pode mantê-los como paridade com a referência (`<FUTO0>`, `<FUTO1>`, ...), ou substituir por palavras de alta frequência da sua língua. A referência tem contrações em inglês nos slots de conteúdo; você poria as palavras/contrações frequentes do seu idioma ali.

<a name="33-the-keypress-prompt-format-not-literal-text"></a>
### 3.3 O formato de prompt por keypress (NÃO é texto literal)

Esse é o **detalhe não-documentado mais importante de todos**. O snippet do wiki mostra ele, mas é fácil ler errado:

> `This is some <XBU><CHAR_T><CHAR_X><CHAR_E><CHAR_T><XBC>text <XEC>`

O modelo **não** é prompt-ado com `<XBU>typo<XBC>`. Ele é prompt-ado com a sequência de teclas digitadas como tokens `<CHAR_X>` discretos. Cada caractere que o usuário digita vira um token `<CHAR_<UPPER>>`, e o modelo é treinado pra prever a palavra corrigida como texto puro entre `<XBC>` e `<XEC>`.

Certo (verificado, ~74% de acurácia top-1 na referência em inglês):
```
prompt:  <XBU><CHAR_T><CHAR_E><CHAR_H><XBC>
output:  The <XEC>...
```

Errado (0% de acurácia, modelo produz besteira tipo "I" e depois `<XEC>` e desiste):
```
prompt:  <XBU>teh<XBC>
```

#### Implicação pra idiomas com diacríticos

O teclado Android manda um token `<CHAR_X>` por **toque físico de tecla**. Diacríticos não são tokens separados. Pro Português isso significa:

- `ã` digitado como long-press-`a` → emite só `<CHAR_A>` (o diacrítico NÃO é preservado no prompt)
- `ç` → `<CHAR_C>`
- `é` → `<CHAR_E>`

Concretamente, a conversão de string de typo pra keypress é:

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

Isso significa que *acento faltando* é a classe de typo dominante pra idiomas ricos em diacríticos — o modelo tem que recuperar acento toda hora. É exatamente pra isso que a autocorreção do FUTO foi feita.

Pra idiomas fora do Latim-26 (Cirílico, Grego, Árabe, CJK), você teria que estender o conjunto `<CHAR_*>`. Isso é arriscado porque o C++ assume um alfabeto Latim de 26 letras (`LETTERS_TO_IDS[0..25]`). Adicionar mais tokens `<CHAR_*>` não vai ser captado; o mapeamento keypress-pra-token no teclado Android precisaria ser patchado. Isso está fora do escopo desse guia.

<a name="34-gguf-metadata-fields"></a>
### 3.4 Campos de metadados GGUF

O namespace `keyboardlm.*` no GGUF guarda metadados específicos do FUTO. Reproduza esses exatamente:

| Campo | Tipo | Valor obrigatório (padrão) |
|---|---|---|
| `keyboardlm.languages` | STRING | ex.: `"pt-BR"` (BCP-47) |
| `keyboardlm.finetuning_count` | UINT32 | `0` pra um modelo novo |
| `keyboardlm.history` | STRING | livre, ex.: `"2026-04-29: Model created"` |
| `keyboardlm.features` | STRING | flags de feature separadas por espaço (veja abaixo) |
| `keyboardlm.ext_tokenizer_type` | STRING | `"sentencepiece"` |
| `keyboardlm.ext_tokenizer_data` | **`[UINT8]`** | os bytes brutos do seu arquivo `.spm` (veja a Armadilha 2 abaixo) |

**A string de `features` é crítica.** O mínimo que funciona pra autocorreção é:

```
base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1
```

O wiki documenta as três primeiras mas **não a quarta**. Sem `char_embed_mixing_v1`, o C++ em `LanguageModel.cpp:94` pula o preenchimento de `LlamaAdapter::embeddings`, e `DecodePromptAndMixes` (que roda a cada keypress) faz

```cpp
float *src = llamaAdapter->embeddings.data() + (t.token * n_embd);
for (size_t i = 0; i < n_embd; i++) mix_f[i] += src[i] * weight;  // SIGSEGV
```

contra um vector vazio. A gente descobriu isso puxando o crash backtrace via ADB wireless — veja [Phase 11](#11-side-loading-and-reproducing-a-crash). Na prática você sempre deve declarar `char_embed_mixing_v1` se declarar `xbu_char_autocorrect_v1`. As duas features não são realmente independentes na build atual do FUTO, independente de como o wiki apresenta.

A referência em inglês também declara `xc0_swipe_typing_v1` (ML de entrada por swipe) e exigiria um tensor extra de encoder. Você pode omitir as duas sem problema; o teclado faz fallback pra swipe baseado em dicionário.

---

<a name="4-pipeline-overview"></a>
## 4. Visão geral da pipeline

```
                             [Phase 5 — empacotar & side-load]
                                          ▲
                                          │
[corpus] -- Phase 2 --> [tokenizer] -- Phase 3 --> [pretrain base]
                                                           │
                                                  Phase 4 (3 stages)
                                                           │
                                                   [checkpoint HF final]
                                                           │
                                                   converter + patch
                                                           │
                                                   [arquivo .gguf v2]
                                                           │
                                                       celular
```

Cada phase tem seu próprio script nesse projeto em `scripts/0N_*.py`. Rode em ordem; a saída da phase N é entrada da phase N+1.

---

<a name="5-phase-0--extract-the-official-model-as-your-spec"></a>
## 5. Phase 0 — extrair o modelo oficial como sua spec

Antes de qualquer outra coisa, pega a referência e despeja:

```bash
mkdir my-keyboard && cd my-keyboard
python -m venv env && source env/bin/activate
pip install sentencepiece huggingface_hub gguf protobuf numpy

git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

hf download breadlicker45/futo-keyboard-lm --local-dir reference_model/
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py \
       reference_model/ml4_1_f16_meta_fixed.gguf > reference_metadata.txt
```

Trate `reference_metadata.txt` como sua spec. Cada campo de metadado, cada constante arquitetural, cada nome de tensor — bata com ele.

Você também quer o SentencePiece embarcado extraído de dentro do GGUF, porque ele te diz quais strings de símbolo especial realmente aparecem:

```python
from gguf import GGUFReader
r = GGUFReader("reference_model/ml4_1_f16_meta_fixed.gguf")
spm_field = r.fields["keyboardlm.ext_tokenizer_data"]
spm_bytes = bytes(int(spm_field.parts[i].tolist()[0]) for i in spm_field.data)
open("reference_model/extracted_spm.model", "wb").write(spm_bytes)
```

Abre `extracted_spm.model` com `sentencepiece` e imprime as peças 4..303 — esse é o mapa de slots.

---

<a name="6-phase-1--assemble-a-target-language-corpus"></a>
## 6. Phase 1 — montar um corpus no idioma alvo

Pra um teclado de celular, **registro importa mais que volume**. Um corpus de 5B tokens de texto formal estilo Wikipedia te dá um modelo péssimo em prever `vc tô indo agora` e ótimo em prever `the diaspora demographically referred to`. As pessoas digitam o primeiro tipo, não o segundo.

O mix que funcionou na nossa rodada pt-BR:

| Estilo da fonte | Dataset HF (abril de 2026) | Notas |
|---|---|---|
| Web crawl, idioma alvo | (varia — `eduagarcia/BrWac` pra pt-BR) | Volume; estatística abrangente |
| Wikipedia | `wikimedia/wikipedia` config `<dateid>.<lang>` | Fundamentação formal |
| **Legendas / diálogo de cinema** | `Helsinki-NLP/opus-100` config `<en-yourlang>` | **O sinal mais forte de registro casual** |

A gente tentou `mozilla-foundation/common_voice_*` e `gabrielrstan/CORAA-v1.1` pra ter mais dado conversacional. **Os dois falharam no início de 2026**: o Common Voice exige auth ou foi reestruturado, e o layout de dados do CORAA não bate com o auto-loader do HF. Se você está lendo isso depois, tenta de novo — pode ter voltado. Se não, scrapeia o seu (dumps de fórum/Reddit/Twitter).

Streaming do HF é inegociável nessa escala. Não tenta baixar o BrWaC ou OSCAR inteiro — vai acabar o disco. Stream, filtra, escreve arquivos de texto shardeados. O script `scripts/01_build_corpus.py` mostra o padrão; pontos-chave:

- `streaming=True` no `load_dataset`
- filtro por documento: tamanho mínimo (menor pra legendas), descartar ruído, filtro de idioma pra corpora mistos
- dedup por hash dos primeiros ~200 chars
- escrever em arquivos `shard_NNNNN.txt` de ~256 MiB cada
- monitorar a contagem aproximada de tokens e parar num budget

Pra uma rodada mini, **500M tokens basta** pra validar a pipeline. Pra uma rodada real, mira em **3-5B tokens** misturados entre registros.

---

<a name="7-phase-2--train-the-sentencepiece-tokenizer"></a>
## 7. Phase 2 — treinar o tokenizer SentencePiece

Você está treinando um BPE de vocab 15008 com os 300 símbolos user-defined fixados nos índices 4..303. A chamada mínima de treino:

```python
import sentencepiece as spm

USER_DEFINED = build_300_symbols()  # veja o script do projeto pro layout dos slots

spm.SentencePieceTrainer.train(
    input=",".join(corpus_shards),
    input_format="text",
    model_prefix="tokenizer/spm_<lang>",
    vocab_size=15008,
    character_coverage=0.9995,
    model_type="bpe",
    treat_whitespace_as_suffix=True,   # a feature "inverted_space"
    user_defined_symbols=USER_DEFINED, # ganha IDs 4..303 na ordem de declaração
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    byte_fallback=True,
    input_sentence_size=2_000_000,
    shuffle_input_sentence=True,
)
```

Valida na hora:

```python
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/spm_<lang>.model")
assert sp.get_piece_size() == 15008
char_ids = [sp.piece_to_id(f"<CHAR_{c}>") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
assert char_ids == list(range(char_ids[0], char_ids[0]+26)), \
    "<CHAR_A>..<CHAR_Z> must be sequential"
assert sp.piece_to_id("<XBU>") > 3
```

Se o assert falhar em `<CHAR_*>`, sua lista `user_defined_symbols` não está sequencial. Arruma a ordem.

### A grande armadilha de RAM

BPE de SentencePiece é gargalo de memória, não de compute. A fase de merge mantém o conjunto inteiro de treino + uma hash table de contagens de pares residentes, e fica perseguindo ponteiros random nessa estrutura. Dá grosseiramente **5-10× o tamanho da entrada em RAM**.

Na nossa rodada, treinar num corpus de 2,0 GiB consumiu 10,6 GiB de RAM, usou um único core de CPU a ~10%, e rodou por **mais de 5 horas num host de 16 GiB que começou a swapar pra disco**. O mesmo treino num host de 31 GiB com RAM DDR4-4000 (sem pressão de swap) terminou em **14 minutos**. Mesmo algoritmo. Mesmos dados. A única diferença foi a folga de memória.

Se o host de treino tem < 32 GiB de RAM, faz uma destas:

1. Pré-amostra o corpus pra ~1-2 GiB antes de treinar: `shuf shards/*.txt | head -c 1500000000 > sp_sample.txt`
2. Roda o treino do tokenizer numa máquina com mais RAM e copia o `.spm`.
3. Adiciona 32+ GiB de swap como rede de segurança (lento mas não vai OOM-killar).

Esse é o tipo de coisa que ninguém te avisa de antemão. A gente quase matou uma rodada porque achou que o algoritmo tinha travado.

---

<a name="8-phase-3--pretrain-the-base-model"></a>
## 8. Phase 3 — pretrain do modelo base

Esse é um loop padrão de Trainer do HuggingFace com o `LlamaConfig` verificado. Decisões-chave:

```python
TrainingArguments(
    max_steps=100_000,                  # pra rodada real; 20k pra validação mini
    per_device_train_batch_size=16,     # micro batch
    gradient_accumulation_steps=16,     # → batch global 256
    learning_rate=3e-4,
    warmup_steps=2000,
    lr_scheduler_type="cosine",
    bf16=True,
    weight_decay=0.1,
    save_steps=5000,
)
```

Sequence length de 1024 cabe confortavelmente em 24 GiB com micro-batch 16. Com seq=1024 × batch 256 = ~262k tokens por passo. 100k passos × 262k = ~26B tokens de treino. Num corpus de 5B tokens isso é uns 5 epochs.

Faz streaming do corpus em vez de carregar tudo. Tokeniza on the fly e empacota num tamanho fixo. Os workers são gargalo de I/O, não de GPU — 2-4 workers basta.

**Salva a cada 2500-5000 passos** pra poder retomar de qualquer checkpoint sem perder mais que algumas horas.

Espere uma cross-entropy loss final em torno de 3-4 numa rodada de qualidade real, perplexidade em torno de 30-50. A referência em inglês provavelmente tinha perplexidade em torno de 25-30. Você vai ver a sua loss platear lá pelos 60-80% do total de passos; o resto é refinar o vocabulário de cauda longa.

### Sanity check da rodada mini

A gente fez um pretrain de 20k passos em 500M tokens só de Wikipedia-pt. Loss final 3,81, perplexidade ~45. Carregou normal, rodou normal, mas as sugestões eram ruins. Não espera que uma rodada de 20k passos só com Wikipedia produza um modelo de teclado usável — é smoke test, nada mais.

---

<a name="9-phase-4--autocorrect-fine-tune-3-stages"></a>
## 9. Phase 4 — fine-tune de autocorreção (3 stages)

O modelo de pretrain não tem ideia do que significa o formato `<XBU>...<XEC>`. A Phase 4 ensina. Três stages sequenciais, cada um carregando do anterior:

### Phase 4a: triplas de autocorreção isoladas

Gera um dataset sintético de exemplos `<XBU><CHAR_*>...<XBC>correct<XEC>`, amostrando palavras de um mapa de frequência do seu corpus. Estratégias pro lado da string de typo:

- Cortar acentos (a classe de typo dominante pra idiomas com diacrítico: `voce`→`você`, `nao`→`não`)
- Perder cedilha: `coracao`→`coração`
- Typos de adjacência no teclado (layout qwerty)
- Caracteres transpostos/duplicados/faltando
- Atalhos comuns que seus falantes realmente digitam (`vc`→`você`, `tb`→`também` em pt-BR)

Pondera a amostragem por `log(word_freq + 1)` pra palavras comuns não afogarem as raras (é o que o wiki do FUTO recomenda, e é razoável).

Treina com **labels mascarados em tudo, exceto no span da correção** (entre `<XBC>` e `<XEC>`):

```python
labels = [-100] * len(input_ids)
xbc_pos = input_ids.index(sp.piece_to_id("<XBC>"))
for k in range(xbc_pos, len(input_ids)):
    labels[k] = input_ids[k]
```

Isso foca a loss no que o modelo precisa de fato *prever*. Cerca de 5-10K passos com seq_len=64, batch 256, lr 1e-4 basta — ir mais longe faz overfit na distribuição sintética.

### Phase 4b: autocorreção em contexto

Pega frases reais do seu corpus, substitui aleatoriamente ~20-30% das palavras por triplas `<XBU><CHAR_*>...<XBC>correct<XEC>` no lugar. A loss agora computa sobre todos os tokens (sem mask), então o modelo aprende tanto autocorreção quanto modelagem contínua de linguagem.

```
Eu fui ao <XBU><CHAR_M><CHAR_E><CHAR_C><CHAR_A><CHAR_D><CHAR_O><XBC>mercado<XEC> ontem
```

Treina por 20-30K passos com seq_len=512, lr 5e-5. Cuidado com a taxa de typo — alta demais (>40%) e o modelo entra em mode collapse pra "sempre emitir uma tripla". A gente viu isso com taxa 0,33 na nossa rodada; 0,20-0,25 é mais seguro.

### Phase 4c: adaptação conversacional

Esse stage **não** está no wiki do FUTO. A gente adicionou depois de observar que até a saída da Phase 4b soava como Wikipedia. Pega um corpus menor de texto casual na sua língua (legendas, chat, etc., não web crawl formal), aplica a injeção de typo da Phase 4b numa taxa *menor* (~0,10), e faz um fine-tune curto (~5K passos, lr 2e-5). Isso desloca o modelo na direção do registro de digitação sem apagar a estatística da língua.

---

<a name="10-phase-5--package-as-a-futo-compatible-gguf"></a>
## 10. Phase 5 — empacotar como GGUF compatível com FUTO

Essa phase é curta, frustrante, e onde moram as armadilhas menos óbvias. Cinco coisas precisam estar certas pro arquivo carregar e não crashar:

1. **GGUF versão 2** (a llama.cpp vendorada no FUTO não lê v3+).
2. **Sem campos KV que o parser antigo não reconhece** (~9 campos específicos produzidos pelo `convert_hf_to_gguf.py` recente precisam ser removidos).
3. **`output.weight` quantizado em Q6_K** (bate com a referência; F16 pode funcionar mas bater com a referência é mais seguro).
4. **`keyboardlm.features` inclui `char_embed_mixing_v1`** (caso contrário SIGSEGV no meio da inferência, veja Seção 11).
5. **`keyboardlm.ext_tokenizer_data` é array `[UINT8]`, não `[INT32]`.**

Pipeline concreta:

```bash
# 5.1: monta o checkpoint HF com o SentencePiece como tokenizer.model
mkdir -p staged
cp finetune/stage_c/final/* staged/
cp tokenizer/spm_<lang>.model staged/tokenizer.model
echo '{"tokenizer_class":"LlamaTokenizer","model_max_length":2048,...}' \
     > staged/tokenizer_config.json
echo '{"bos_token":"<s>","eos_token":"</s>","pad_token":"<pad>","unk_token":"<unk>"}' \
     > staged/special_tokens_map.json

# 5.2: HF -> GGUF baunilha via llama.cpp
python llama.cpp/convert_hf_to_gguf.py staged/ \
       --outfile vanilla.gguf --outtype f16

# 5.3: requantiza output.weight pra Q6_K (bate com a referência)
./llama.cpp/build/bin/llama-quantize \
    --allow-requantize \
    --output-tensor-type q6_k \
    vanilla.gguf q6kout.gguf f16

# 5.4: patcheia os metadados FUTO no GGUF
python scripts/06_patch_metadata.py \
    --in q6kout.gguf --out futo_v3.gguf \
    --tokenizer tokenizer/spm_<lang>.model \
    --languages "<lang-tag>" \
    --features "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"

# 5.5: downgrade de GGUF v3 → v2 e remove campos extras
python scripts/06b_downgrade_v2.py --in futo_v3.gguf --out futo_v2_final.gguf
```

O passo `06_patch_metadata.py` tem uma exigência sutil: **embarcar os bytes do SentencePiece**. O óbvio `writer.add_array(name, list(spm_bytes))` produz um array `[INT32]`, que o C++ do FUTO rejeita. Use `add_key_value` com `sub_type` explícito:

```python
from gguf import GGUFValueType
spm_bytes = open("tokenizer.model", "rb").read()
writer.add_key_value(
    "keyboardlm.ext_tokenizer_data",
    spm_bytes,                         # bytes, não list[int]
    GGUFValueType.ARRAY,
    sub_type=GGUFValueType.UINT8,      # crítico
)
```

O passo de downgrade (`06b_downgrade_v2.py`) lê o GGUF, copia todos os campos exceto uma deny-list de 9 campos mais novos que v2 (`general.size_label`, `general.type`, `llama.attention.key_length`, `llama.attention.value_length`, `llama.vocab_size`, `tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token`, `tokenizer.ggml.padding_token_id`, `tokenizer.ggml.pre`), escreve um GGUF novo, e patcheia o byte de versão no offset 4-7 de `\x03\x00\x00\x00` pra `\x02\x00\x00\x00`.

Depois de tudo isso, o seu `futo_v2_final.gguf` deve ter ~62 MB e exatamente **28 campos KV** batendo com o layout da referência em inglês. Faz um diff final:

```bash
python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py futo_v2_final.gguf > ours.txt
diff <(grep -oP "(?<=\| )[a-z._]+(?= = )" reference_metadata.txt | sort) \
     <(grep -oP "(?<=\| )[a-z._]+(?= = )" ours.txt | sort)
```

As únicas diferenças devem ser as strings `keyboardlm.languages` e `keyboardlm.history` (sua língua vs `'en'`, sua data vs `'2023-11-11'`).

---

<a name="11-side-loading-and-reproducing-a-crash"></a>
## 11. Side-loading e reprodução de um crash

Transfere o `.gguf` pro dispositivo — Syncthing, Nextcloud, USB, ou `adb push /sdcard/Download/`. Daí no FUTO Keyboard: **Languages & Models → Add Model → seleciona seu arquivo → atribui pra sua língua → Text Prediction → habilita Transformer LM**.

Se carregar como "(Unsupported)": abre o dump do GGUF e checa se todos os seis campos `keyboardlm.*` estão presentes e se a versão é 2.

Se o teclado abre normal mas **fecha no momento em que você digita 2-3 letras**: é crash nativo (SIGSEGV), quase sempre um destes:

- Falta a feature `char_embed_mixing_v1`
- `keyboardlm.ext_tokenizer_data` é INT32 em vez de UINT8
- Algum dos special tokens estruturais (`<XBU>`/`<XBC>`/`<XEC>`/`<XC0>`/`<CHAR_A>`) não entrou no seu SentencePiece

Pra debugar, habilita ADB wireless no celular (Developer Options → Wireless Debugging → Pair device with code), aí da sua máquina Linux:

```bash
adb pair <IP>:<PAIRING_PORT> <CODE>     # uma vez só
adb connect <IP>:<DEBUG_PORT>            # cada sessão
adb logcat -c                            # limpa buffer
adb logcat | grep -E "Fatal signal|F libc|F DEBUG|appDiedLocked.*futo"
```

Reproduz o crash. Você vai ver algo do tipo:

```
F libc    : Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x64800
            in tid NNN (LanguageModel)
F DEBUG   : signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000000000064800
F DEBUG   : backtrace:
F DEBUG   :   #00 pc 0x1681ac libjni_latinime.so
              (LanguageModelState::DecodePromptAndMixes(...) + 1904)
```

Um endereço de falha pequeno (abaixo de 0x100000) significa um dereference de `nullptr + offset`, quase sempre uma feature faltando ou campo não inicializado. O nome da função e o offset apontam pra exatamente qual membro está não inicializado — leia `native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp` perto do offset, e `native/jni/src/ggml/LanguageModel.cpp` pra inicialização do `LlamaAdapter`.

Se você vê `ASSERT failed` cedo no start do app (antes de digitar), tem um token estrutural faltando no tokenizer. Os asserts estão em `LanguageModel.cpp:222-225` e 233.

---

<a name="12-the-five-gotchas-in-one-place"></a>
## 12. As cinco armadilhas, num lugar só

Em ordem da "mais provável de morder um iniciante" pra menos:

1. **O formato de keypress é com tokens `<CHAR_X>`, não texto literal.** O snippet do wiki insinua mas é fácil ler errado. Verificar com um prompt de teste contra a referência em inglês leva 30 segundos e poupa dias.
2. **`char_embed_mixing_v1` é obrigatório se você declara `xbu_char_autocorrect_v1`.** O wiki apresenta as duas como features independentes. Não são. Sem `char_embed_mixing_v1`, cada keystroke dá SIGSEGV.
3. **GGUF tem que ser versão 2.** O `convert_hf_to_gguf.py` mais novo produz v3 com campos KV extras que a llama.cpp vendorada do FUTO não parseia. Você precisa de um passo de downgrade.
4. **`keyboardlm.ext_tokenizer_data` precisa ser `[UINT8]`.** O óbvio `add_array(name, list(bytes))` produz `[INT32]` (4× o tamanho, layout errado). Use `add_key_value(..., sub_type=GGUFValueType.UINT8)`.
5. **`<CHAR_A>..<CHAR_Z>` precisam ser 26 IDs de token contíguos e sequenciais.** O C++ faz aritmética de ponteiros no ID de `<CHAR_A>`. Listar eles em ordem em `user_defined_symbols` basta.

Existem coisas menores também — `<CHAR_A>=182` e `<XBU>=174` na referência (a gente manteve os mesmos índices por segurança, mesmo que a maioria seja resolvida pelo nome); `output.weight` é Q6_K em vez de F16 na referência; o SentencePiece no GGUF é o tokenizer *de verdade* que o C++ usa, não o array `tokenizer.ggml.tokens`. Nenhuma dessas é fatal sozinha, mas juntas dá pra perder um fim de semana se você não souber checar.

---

<a name="13-evaluation-methodology"></a>
## 13. Metodologia de avaliação

Antes de jogar num celular, avalia contra o mesmo conjunto de testes em três checkpoints:

| Referência | O que representa |
|---|---|
| **Teto** — modelo oficial em inglês rodando em testes em inglês | Qualidade no melhor cenário com essa arquitetura e escala de dados |
| **Piso** — seu checkpoint de pretrain *antes* da Phase 4 | Testa autocorreção num modelo que não viu o formato. Deve ficar perto de 0%. |
| **Final** — seu modelo pós-Phase-4 | Deve ficar bem acima do piso; idealmente perto do teto. |

Pra testes de autocorreção, use ~30 exemplos por idioma cobrindo categorias: atalhos, acento faltando, transposto, duplicado, erros comuns de escrita. Pra cada um: alimenta `<XBU><CHAR_*>...<XBC>` e checa se o modelo emite a palavra correta antes do `<XEC>`. Acompanha top-1 (greedy) e top-5 (sampled).

A referência em inglês marca ~74% top-1 / 89% top-5 num conjunto de 27 perguntas de autocorreção em inglês do tipo descrito acima. Uma rodada bem-sucedida pt-BR deve mirar números semelhantes num conjunto paralelo em pt-BR.

Um modo de falha pra observar: **final ≈ piso**. Isso significa que a Phase 4 não ensinou o formato pro modelo. Confere (a) que o dataset realmente usa o formato de keypress, (b) que o mascaramento de label na Phase 4a está certo, (c) que a loss caiu de forma substancial durante a Phase 4. Se sua loss final depois da Phase 4a estiver >1,0, o fine-tune provavelmente não convergiu.

---

<a name="14-what-this-guide-does-not-cover"></a>
## 14. O que esse guia não cobre

- **Tuning de hiperparâmetro além de um baseline funcional.** Os configs acima produzem *um* modelo funcional; chegar perto da qualidade do teto é um esforço próprio de várias semanas.
- **Quantização além de Q6_K no output.** A referência é majoritariamente F16 com output Q6_K; quantizações menores (Q4_K_M etc.) reduzem o tamanho do arquivo mas a gente não testou a compatibilidade com o loader do FUTO.
- **Fine-tune LoRA on-device.** A feature `lora_finetunable_v1` existe na spec do FUTO mas a gente não habilitou. Adicionar exige preparar o modelo com metadados de tensor específicos; fora do escopo aqui.
- **ML de swipe (`xc0_swipe_typing_v1` + `experiment_linear_208_209_210`).** Exige tensores de encoder adicionais nos índices hard-coded 208/209/210. Pula; o swipe clássico do FUTO continua funcionando sem ML e o teclado segue usável.
- **Idiomas fora do Latim-26.** Os 26 slots `<CHAR_*>` hardcoded assumem um alfabeto de 26 caracteres. Cirílico, Grego, Árabe, etc. exigiriam patches no próprio teclado Android do FUTO — não só um tokenizer diferente.
- **Deploy em produção.** Esse é um guia de side-loading. Distribuir seu modelo como pacote instalável ou contribuir ele de volta upstream pro FUTO é outra conversa.

---

<a name="15-acknowledgements-and-licensing"></a>
## 15. Agradecimentos e licenciamento

Esse guia foi escrito de forma independente. Ele documenta comportamento não-documentado do app Android do FUTO Keyboard (open source, veja o repo do FUTO pra licença dele) por engenharia reversa de crashes, leitura do source C++, e verificação de comportamento via inferência real. Os detalhes arquiteturais e de metadado são fatos sobre como o app funciona; não são obra criativa de propriedade do FUTO.

A referência em inglês `breadlicker45/futo-keyboard-lm` é um re-upload do modelo oficial em inglês do teclado do FUTO. A gente usa a estrutura byte-level dele como alvo de especificação; não redistribuímos.

Os artefatos concretos do projeto (scripts de treino, patcher de metadados GGUF, script de downgrade) estão sob licença MIT — adapta livre pra sua língua. Os pesos do modelo que você produz são seus.

Se o FUTO publicar scripts oficiais de treino multi-idioma, prefira esses ao invés desse guia. Eles disseram que pretendem revisitar a pipeline de ML; se você fizer esse trabalho, seu modelo pode precisar ser re-empacotado quando isso acontecer. Trate qualquer modelo que você shipa como um investimento "bom pelos próximos 6-18 meses", não uma solução permanente.

O jeito mais rápido de validar as alegações do wiki do FUTO é o mesmo que a gente usou: treinar, empacotar, side-loadar, e ler o crash. Várias descobertas desse guia vieram de um SIGSEGV no offset 0x64800 em `DecodePromptAndMixes` e da meia tarde de mergulho no source que veio depois. A spec de formato emerge fazendo o trabalho — não lendo.

Divirta-se, e boa sorte.
