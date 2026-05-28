"""
Generate a diverse pt-BR chat-style corpus via `claude -p` for v8.2-base
continue-pretraining (Wu et al. 2024 recipe — +22.8% NWP improvement reported
on Gboard from LLM-prompted diverse chat data as continue-pretrain).

Output: a single flat text file (one message per line), suitable for the
existing PtBrShardStreamer (scripts/03_pretrain.py) and `09_continue_pretrain_synth.py`.

The shape: ~50 personas × ~15 topics × ~15 messages per call = ~10K messages.
Reasonable wall: ~25 min at concurrency 8 (~12s per `claude -p` call).

Usage:
  python3 scripts/gen_synth_corpus.py \\
      --out corpora/synth_v82/shard_00000.txt \\
      --per-cell 4 \\
      --concurrency 8 \\
      --model claude-haiku-4-5
"""
from __future__ import annotations
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Personas — crossed with topics to drive diversity. ~50 personas covering
# common pt-BR speaker types.
PERSONAS = [
    "uma adolescente de 16 anos de São Paulo, fala gíria, usa kkkk",
    "um homem de 35 anos engenheiro de software de Belo Horizonte",
    "uma mulher de 28 anos professora de português do Rio",
    "um aposentado de 67 anos de Curitiba, mais formal",
    "uma estudante universitária de 21 anos de Salvador",
    "um motoboy de 24 anos de São Paulo",
    "uma médica de 40 anos de Porto Alegre",
    "um adolescente de 14 anos gamer, fala em gíria de jogo",
    "uma mãe de 38 anos, dona de casa, do interior de Minas",
    "um pai de 45 anos, advogado, de Brasília",
    "uma jornalista de 32 anos de Recife",
    "um cozinheiro de 29 anos, paulista, gosta de gíria",
    "uma cantora de 26 anos de Belo Horizonte",
    "um músico de 33 anos, baiano, descontraído",
    "uma corretora de imóveis de 41 anos de Florianópolis",
    "um vendedor de carros de 36 anos de Goiânia",
    "uma psicóloga de 31 anos, gaúcha",
    "um caminhoneiro de 50 anos do interior de São Paulo",
    "uma confeiteira de 27 anos de Belém",
    "um arquiteto de 39 anos paulistano",
    "uma estilista de 30 anos carioca, descolada",
    "um youtuber de 22 anos de Manaus",
    "uma fisioterapeuta de 33 anos de Fortaleza",
    "um eletricista de 44 anos do interior do Paraná",
    "uma manicure de 26 anos do interior de São Paulo",
    "um barbeiro de 28 anos de Salvador",
    "uma vendedora de loja de roupas de 23 anos de Recife",
    "um técnico em informática de 30 anos de Vitória",
    "uma bióloga de 34 anos pesquisadora de Manaus",
    "um chefe de cozinha de 42 anos paulistano, sofisticado",
    "uma jovem mãe de 25 anos, primeiro filho, do Rio Grande do Sul",
    "um pai coruja de 38 anos, dois filhos pequenos",
    "uma adolescente de 17 anos, fã de K-pop, de São Paulo",
    "um universitário de 19 anos, calouro de medicina",
    "uma trabalhadora autônoma de 35 anos, freelancer de design",
    "um padre de 55 anos do interior de Pernambuco",
    "uma pastora de 47 anos de São Paulo",
    "um historiador de 50 anos, professor universitário",
    "uma escritora de 38 anos, descolada, do Rio",
    "um piloto de avião de 45 anos de Brasília",
    "uma comissária de bordo de 30 anos de São Paulo",
    "um analista financeiro de 32 anos paulistano",
    "uma social media de 25 anos, fala em gíria de internet",
    "um personal trainer de 29 anos carioca",
    "uma nutricionista de 33 anos de Florianópolis",
    "um veterinário de 36 anos do interior",
    "uma dentista de 41 anos de Salvador",
    "um professor de matemática de 50 anos descontraído",
    "uma garçonete de 22 anos de São Paulo",
    "um pedreiro de 48 anos do nordeste",
]


# Reuse the same 13 scenarios from gen_conversations.py for parity, plus a few
# extras for breadth (since this is training data, not eval).
TOPICS = [
    ("daily_logistics",      "fazendo planos do dia: mercado, contas, busca na escola, transporte"),
    ("food_breakfast",       "café da manhã: pão, ovo, fruta, café, leite, padaria, opções"),
    ("food_lunch",           "almoço: prato, restaurante, marmita, sobremesa, comida de boteco"),
    ("food_dinner",          "jantar: receita, ingredientes, delivery, pedidos, prato do dia"),
    ("movies_opinion",       "opinião sobre filmes recém-lançados, séries, atores, diretores"),
    ("social_media_opinion", "opinião sobre redes sociais: Instagram, TikTok, Twitter, influencers"),
    ("celebrities_opinion",  "opinião sobre celebridades brasileiras e internacionais, fofocas"),
    ("news_opinion",         "opinião sobre notícias atuais: política, economia, sociedade, esporte"),
    ("philosophy",           "discussão filosófica leve: sentido da vida, ética, felicidade"),
    ("friends_casual",       "papo casual com amigos, marcando rolê, café, cerveja"),
    ("parenting",            "conversa de pais: escola, saúde dos filhos, comportamento, dever de casa"),
    ("gig_planning",         "organizando um show ou rolê musical: local, horário, lineup, transporte"),
    ("workmates",            "colegas de trabalho: projetos, prazos, reuniões, fofoca de escritório"),
    ("travel",               "planos de viagem: destino, passagem, hotel, roteiro"),
    ("health_fitness",       "saúde e exercício: academia, dieta, sono, esporte"),
    ("hobby_books",          "livros lidos recentemente, autores favoritos, indicações"),
    ("hobby_music",          "música favorita, artistas, shows, festivais, playlists"),
    ("home_repairs",         "obras em casa, reformas, encanador, eletricista, marceneiro"),
    ("relationships",        "relacionamento: encontros, namoro, casamento, dicas, problemas"),
    ("weather_chat",         "tempo, calor, chuva, frio, previsão, roupa apropriada"),
]


CORPUS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 10,
            "maxItems": 20,
        },
    },
    "required": ["messages"],
})


PROMPT = """\
Você é {persona}.

Escreva 12-18 mensagens curtas e naturais que essa pessoa enviaria em conversas casuais sobre o tema abaixo. Cada mensagem é INDEPENDENTE — não é uma conversa contínua, é uma coleção de mensagens que essa pessoa poderia escrever em diferentes momentos.

Tema: {topic}

Regras:
- Cada mensagem com 5 a 18 palavras.
- Estilo de mensagem de WhatsApp/SMS — coloquial, brasileiro, NATURAL.
- Use contrações ("tô", "tá", "vc", "pra", "né", "kkkk") quando combinar com a persona.
- A grafia deve estar predominantemente CORRETA (com acentos e cedilhas) — esta é a versão final corrigida.
- Variedade: misture afirmações, perguntas, exclamações, comentários, opiniões.
- Sem emoji, sem markdown, sem prefixos.
- Mensagens DIVERSAS — evite repetir estrutura ou começos.

Saída: APENAS um objeto JSON com chave "messages" (lista de strings).
"""


def run_claude(prompt: str, model: str, schema: str, timeout: int = 90) -> dict:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json",
           "--no-session-persistence", "--json-schema", schema, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"claude exit {res.returncode}: {res.stderr[:200]}")
    env = json.loads(res.stdout)
    if env.get("is_error"):
        raise RuntimeError(f"claude error: {str(env.get('result', ''))[:200]}")
    out = env.get("structured_output")
    if not out:
        txt = (env.get("result") or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
        out = json.loads(cleaned)
    return out


def gen_one(persona: str, topic_key: str, topic_desc: str, model: str) -> tuple[str, list[str] | None, str]:
    prompt = PROMPT.format(persona=persona, topic=topic_desc)
    for attempt in range(3):
        try:
            obj = run_claude(prompt, model, CORPUS_SCHEMA)
            msgs = obj.get("messages", [])
            cleaned = []
            for m in msgs:
                if not isinstance(m, str): continue
                m = m.strip()
                w = m.split()
                if 3 <= len(w) <= 25 and "<" not in m:  # filter format errors
                    cleaned.append(m)
            if len(cleaned) < 5:
                raise RuntimeError(f"too few clean messages: {len(cleaned)}")
            return ("ok", cleaned, "")
        except Exception as e:
            if attempt == 2:
                return ("err", None, f"{type(e).__name__}: {e}")
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output text file (one message per line).")
    ap.add_argument("--per-cell", type=int, default=4,
                    help="Calls per persona-topic cell. 1 call ≈ 12-18 messages.")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=None,
                    help="Cap total calls (for smoke tests).")
    ap.add_argument("--persona-sample", type=int, default=None,
                    help="Sample N personas instead of all (for smoke tests).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not shutil.which("claude"):
        sys.exit("`claude` CLI not on PATH.")

    rng = random.Random(args.seed)
    personas = PERSONAS
    if args.persona_sample:
        personas = rng.sample(personas, min(args.persona_sample, len(personas)))

    work = []
    for p in personas:
        for tk, td in TOPICS:
            for _ in range(args.per_cell):
                work.append((p, tk, td))
    rng.shuffle(work)
    if args.max_calls:
        work = work[:args.max_calls]

    print(f"Total calls planned: {len(work)} "
          f"({len(personas)} personas × {len(TOPICS)} topics × {args.per_cell} per-cell)"
          f" — concurrency={args.concurrency}, model={args.model}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    f = open(args.out, "w", encoding="utf-8")

    accepted_msgs = 0
    accepted_calls = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(gen_one, p, tk, td, args.model): (p[:30], tk) for p, tk, td in work}
        for fut in as_completed(futures):
            status, msgs, err = fut.result()
            if status == "ok":
                for m in msgs:
                    f.write(m + "\n")
                    accepted_msgs += 1
                accepted_calls += 1
                f.flush()
                if accepted_calls % 20 == 0:
                    elapsed = time.time() - t0
                    rate = accepted_msgs / max(elapsed, 1)
                    print(f"  calls={accepted_calls}/{len(work)} msgs={accepted_msgs} "
                          f"err={errors} {rate:.1f} msg/s", file=sys.stderr)
            else:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR ({futures[fut]}): {err}", file=sys.stderr)

    f.close()
    elapsed = time.time() - t0
    print(f"\nWrote {accepted_msgs} messages from {accepted_calls} calls "
          f"({errors} errors) in {elapsed/60:.1f} min to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
