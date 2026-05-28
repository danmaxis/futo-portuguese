"""
Generate short pt-BR conversations across 13 scenarios via `claude -p`.

Used by scripts/eval_conversational.py to measure NWP + prefix-completion
top-1 / top-3 quality of a candidate model. Complements the typo-correction
eval (eval_real_typos.py) which only measures correction, not flow.

Output: JSON list of {scenario, messages: [{speaker, text}, ...]}.

Usage:
  python3 scripts/gen_conversations.py \\
      --out notes/v8_1/conversations.json \\
      --per-scenario 5 \\
      --concurrency 4 \\
      --model claude-haiku-4-5
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# 13 scenarios — 3 of mine + 10 from the user
SCENARIOS = [
    ("friends_casual",       "amigos batendo papo casual, marcando um café ou rolê"),
    ("family_logistics",     "casal/família resolvendo logística do dia (mercado, conta, busca na escola)"),
    ("workmates",            "colegas de trabalho discutindo um projeto, prazos, tarefas"),
    ("parenting",            "pais conversando sobre os filhos: escola, saúde, comportamento, dever de casa"),
    ("gig_planning",         "amigos organizando um show ou rolê musical: local, horário, lineup, transporte"),
    ("breakfast",            "discutindo o que comer no café da manhã: pão, ovos, fruta, café, leite"),
    ("lunch",                "discutindo o almoço: prato, restaurante, marmita, sobremesa"),
    ("dinner",               "discutindo o jantar: receita, restaurante, ingredientes, pedidos"),
    ("op_movies",            "opinião sobre filmes recém-lançados, atores, diretores, gêneros"),
    ("op_social_media",      "opinião sobre redes sociais: Instagram, TikTok, Twitter, influencers, algoritmo"),
    ("op_celebrities",       "opinião sobre celebridades brasileiras e internacionais, fofocas, carreira"),
    ("op_news",              "opinião sobre notícias atuais: política, economia, sociedade, esporte"),
    ("op_philosophy",        "discussão filosófica leve: sentido da vida, ética, livre-arbítrio, felicidade"),
]


CONV_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
            "minItems": 3,
            "maxItems": 8,
        },
    },
    "required": ["messages"],
})


PROMPT = """\
Gere uma conversa CURTA e CASUAL em português brasileiro entre DUAS pessoas (A e B) sobre o seguinte tema:

Tema: {description}

Regras:
- 4 a 7 mensagens no total, alternando entre A e B.
- Cada mensagem com 5 a 15 palavras.
- Estilo coloquial brasileiro NATURAL (use "vc", "tá", "né", "kkk" às vezes; contrações; gírias leves quando couber).
- A grafia deve estar predominantemente CORRETA (com acentos e cedilhas) — esta é a forma final, já corrigida da mensagem.
- Sem emoji, sem markdown, sem prefixos tipo "A:" — só o texto puro no campo "text".

Saída: APENAS um objeto JSON com a chave "messages", lista de objetos {{speaker, text}}.
"""


def run_claude(prompt: str, model: str, schema: str, timeout: int = 60) -> dict:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json",
           "--no-session-persistence", "--json-schema", schema, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"claude exit {res.returncode}: {res.stderr[:200]}")
    env = json.loads(res.stdout)
    if env.get("is_error"):
        raise RuntimeError(f"claude error: {env.get('result', '')[:200]}")
    out = env.get("structured_output")
    if not out:
        txt = (env.get("result") or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
        out = json.loads(cleaned)
    return out


def gen_one(scenario_key: str, description: str, model: str) -> tuple[str, dict | None, str]:
    prompt = PROMPT.format(description=description)
    for attempt in range(3):
        try:
            obj = run_claude(prompt, model, CONV_SCHEMA)
            msgs = obj.get("messages", [])
            if not (3 <= len(msgs) <= 8):
                raise RuntimeError(f"bad message count: {len(msgs)}")
            for m in msgs:
                if not m.get("text", "").strip():
                    raise RuntimeError("empty text")
            return ("ok", {"scenario": scenario_key, "messages": msgs}, "")
        except Exception as e:
            if attempt == 2:
                return ("err", None, f"{type(e).__name__}: {e}")
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scenario", type=int, default=5)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    if not shutil.which("claude"):
        sys.exit("`claude` CLI not on PATH.")

    work = []
    for key, desc in SCENARIOS:
        for _ in range(args.per_scenario):
            work.append((key, desc))

    print(f"Generating {len(work)} conversations ({len(SCENARIOS)} scenarios × "
          f"{args.per_scenario}) at concurrency={args.concurrency}", file=sys.stderr)

    accepted: list[dict] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(gen_one, k, d, args.model): k for k, d in work}
        for fut in as_completed(futures):
            status, conv, msg = fut.result()
            if status == "ok":
                accepted.append(conv)
                if len(accepted) % 5 == 0:
                    print(f"  accepted={len(accepted)} errors={errors}", file=sys.stderr)
            else:
                errors += 1
                print(f"  ERROR ({futures[fut]}): {msg}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(accepted, ensure_ascii=False, indent=2))

    # Stats
    from collections import Counter
    cnt = Counter(c["scenario"] for c in accepted)
    print(f"\nWrote {len(accepted)} conversations to {args.out}")
    print("By scenario:")
    for k, _ in SCENARIOS:
        print(f"  {k:25s} {cnt[k]}")


if __name__ == "__main__":
    main()
