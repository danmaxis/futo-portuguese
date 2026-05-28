"""
Gemini-flavored synth corpus generator. Runs on the 5070 Ti machine (which has
`gemini` CLI installed). Mirrors gen_synth_corpus.py but shells out to
`gemini -p` instead of `claude -p`. Designed to run in parallel with the
Claude version on a different machine for ~2x throughput.

Default model: gemini-3-flash-preview (the gemini-cli default, fast, good quality).

Usage on 5070 Ti:
  export GEMINI_CLI_TRUST_WORKSPACE=true
  python3 scripts/gen_synth_corpus_gemini.py \\
      --out ~/futo-train/corpora/synth_v82/shard_gemini.txt \\
      --per-cell 2 \\
      --concurrency 8
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Reuse the same personas + topics as gen_synth_corpus.py so the combined
# corpus has parity in coverage. (Inlined here to keep this script standalone
# on the 5070 Ti.)
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


PROMPT = """\
Você é {persona}.

Escreva 12-18 mensagens curtas e naturais que essa pessoa enviaria em conversas casuais sobre o seguinte tema. Cada mensagem é INDEPENDENTE.

Tema: {topic}

Regras:
- Cada mensagem com 5 a 18 palavras.
- Estilo de mensagem de WhatsApp/SMS — coloquial, brasileiro, NATURAL.
- Use contrações ("tô", "tá", "vc", "pra", "né", "kkkk") quando combinar com a persona.
- A grafia predominantemente CORRETA (com acentos e cedilhas).
- Variedade: misture afirmações, perguntas, exclamações, opiniões.
- Sem emoji, sem markdown, sem prefixos.

RESPONDA APENAS com um objeto JSON: {{"messages": ["...", "..."]}}. Sem texto extra antes ou depois.
"""


def run_gemini(prompt: str, model: str | None, timeout: int = 120) -> dict:
    env = os.environ.copy()
    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    cmd = ["gemini", "-p", prompt, "--skip-trust", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"gemini exit {res.returncode}: {res.stderr[:200]}")
    envelope = json.loads(res.stdout)
    raw = (envelope.get("response") or "").strip()
    # The response is JSON text — strip markdown fences if any
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def gen_one(persona: str, topic_key: str, topic_desc: str, model: str | None) -> tuple[str, list[str] | None, str]:
    prompt = PROMPT.format(persona=persona, topic=topic_desc)
    for attempt in range(3):
        try:
            obj = run_gemini(prompt, model)
            msgs = obj.get("messages", [])
            cleaned = []
            for m in msgs:
                if not isinstance(m, str): continue
                m = m.strip()
                w = m.split()
                if 3 <= len(w) <= 25 and "<" not in m:
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--model", default=None,
                    help="Gemini model. None = CLI default (gemini-3-flash-preview).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--persona-sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    if not shutil.which("gemini"):
        sys.exit("`gemini` CLI not on PATH.")

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

    print(f"GEMINI gen: {len(work)} calls "
          f"({len(personas)} personas × {len(TOPICS)} topics × {args.per_cell})"
          f" conc={args.concurrency} model={args.model or 'default'}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    f = open(args.out, "w", encoding="utf-8")

    accepted_msgs = accepted_calls = errors = 0
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
                if accepted_calls % 10 == 0:
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
