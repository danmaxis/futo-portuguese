"""
Generate ~750 high-quality pt-BR typo pairs via the Claude API, using Google's
published two-stage prompt method (research.google/blog/synthetic-and-federated-...).

Stage 1: inject realistic pt-BR errors into clean sentences
Stage 2: re-correct; keep only round-trip-matched pairs (auto QC)

Distribution priors per Gimenes 2015 (aclanthology.org/J15-1011.pdf):
  - 54.9% diacritic errors (omission dominant)
  - Cedilla (ç) has its own class on ABNT2
  - x↔ch, j↔g, s↔z, ç↔ss, e↔i in unstressed, dropped r confusion patterns
  - Most-misspelled words as few-shot anchors

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python scripts/08_claude_synth.py \\
      --seed-corpus corpora/big/shard_00000.txt \\
      --out notes/synth_typos_v3.json \\
      --target-pairs 750 \\
      --model claude-sonnet-4-6
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------- Prompt templates ----------------------------

# Categories per Gimenes 2015 — weights guide the diversity of errors injected.
CATEGORY_PRIORS = {
    "accent_only": 0.40,        # missing diacritics (voce, nao, esta)
    "cedilla_only": 0.10,       # ç → c (coracao, acao)
    "adjacency": 0.20,          # ABNT2 keypress adjacency (merrcado)
    "letter_swap": 0.10,        # x↔ch, j↔g, s↔z, e↔i, dropped r
    "prefix_completion": 0.10,  # user types start, autocompletes wrong (autom → automaticamente)
    "doubled_or_dropped": 0.05, # commer/livvro vs apropriado→apropiado
    "hybrid_multi": 0.05,       # multi-error in one word
}

# Most-misspelled words to use as few-shot anchors
MOST_MISSPELLED = [
    ("excecao", "exceção"), ("escecao", "exceção"), ("beneficente", "beneficente"),
    ("subcidio", "subsídio"), ("previlegio", "privilégio"), ("impecilho", "empecilho"),
    ("apropiado", "apropriado"), ("simplismente", "simplesmente"), ("mecher", "mexer"),
    ("rezolver", "resolver"), ("agradeso", "agradeço"), ("execelente", "excelente"),
    ("voce", "você"), ("nao", "não"), ("esta", "está"), ("tambem", "também"),
    ("coracao", "coração"), ("acao", "ação"), ("estacao", "estação"),
    ("portugues", "português"), ("avo", "avó"),
]


CATEGORY_HINTS = {
    "accent_only": "Bias your error toward a missing-diacritic case (e.g. voce/nao/coracao/esta).",
    "cedilla_only": "Bias your error toward a missing cedilla (ç → c, e.g. coracao/acao).",
    "adjacency": "Bias your error toward ABNT2 keyboard adjacency (a finger-slip to a neighbouring key).",
    "letter_swap": "Bias your error toward an x↔ch, j↔g, s↔z, ç↔ss, or e↔i in unstressed syllable confusion.",
    "prefix_completion": "Bias your error toward a PREFIX of the chosen word — i.e. the user typed only the first 2-6 characters and the keyboard must complete the rest (e.g. 'auto' for 'automaticamente', 'conf' for 'confirma'). 'typed' should be a strict prefix of 'committed', shorter by ≥3 characters.",
    "doubled_or_dropped": "Bias your error toward a doubled or dropped letter (commer/livvro vs apropriado→apropiado).",
    "hybrid_multi": "Bias your error toward a HYBRID multi-error case — combine at least two of (missing diacritic, dropped letter, adjacency slip, doubled letter) in the SAME word, as happens when typing fast on a phone.",
}


STAGE1_PROMPT = """\
You are generating realistic pt-BR (Brazilian Portuguese) misspellings for a mobile keyboard autocorrect training set.

Given a clean Brazilian Portuguese sentence, choose ONE word in it and produce a plausible mistyped form of THAT word. The mistyped form should reflect how Brazilians actually mistype on mobile keyboards, NOT artificial random errors.

{category_hint}

Common Brazilian misspelling patterns (Gimenes 2015, IBM Research):
- ~55% of all pt-BR misspellings are diacritic errors (missing acento or cedilha): "voce" for "você", "nao" for "não", "coracao" for "coração", "esta" for "está"
- ABNT2 keyboard adjacency: ç is right of l on ABNT2; mistyped letters tend to be physically near intended ones
- Confusion of x/ch (mexer→mecher), j/g (especially before e/i), s/z, ç/ss
- e↔i in unstressed syllables: "privilégio"→"previlégio", "simplesmente"→"simplismente"
- Dropped letters (especially r): "apropriado"→"apropiado"
- Brazilian texting shorthand: vc for você, tb for também, hj for hoje, blz for beleza
- Multi-error: in fast typing, multiple errors per word are common

Most commonly misspelled pt-BR words (use as inspiration):
exceção (escecao/excessão), beneficente, subsídio (subcidio), privilégio (previlegio), empecilho (impecilho), apropriado (apropiado), simplesmente (simplismente), mexer (mecher), resolver (rezolver), agradeço (agradeso), excelente (execelente).

You will be given ONE clean sentence. Pick a word in it that is naturally susceptible to one of these error patterns, and output a JSON object with exactly two keys:
- "typed": the misspelled version of the chosen word
- "committed": the correct (original) form of that word

Output ONLY valid JSON. No prose, no markdown, no extra commentary.

Clean sentence:
{sentence}
"""

STAGE2_PROMPT = """\
You are a Brazilian Portuguese spell checker. Given a misspelled pt-BR word, output the most likely intended correct spelling.

Output ONLY the correct word as plain text — no quotes, no JSON, no commentary.

Misspelled word: {typed}
"""


# ---------------------------- Claude CLI helper ----------------------------
# We shell out to `claude -p --output-format json` (the user's Claude Code CLI
# auth, no API key needed). For Stage 1 we pass --json-schema so the model's
# structured_output is a validated JSON object; for Stage 2 we read the plain
# `result` text.

CLAUDE_BIN = shutil.which("claude") or "claude"

STAGE1_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "typed": {"type": "string"},
        "committed": {"type": "string"},
    },
    "required": ["typed", "committed"],
})


def _run_claude(prompt: str, model: str, json_schema: str | None = None,
                timeout: int = 120) -> dict:
    """Invoke `claude -p ...` once and return the parsed JSON envelope."""
    cmd = [CLAUDE_BIN, "-p", "--model", model, "--output-format", "json",
           "--no-session-persistence"]
    if json_schema:
        cmd += ["--json-schema", json_schema]
    cmd.append(prompt)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"claude exit {res.returncode}: {res.stderr[:300]}")
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"non-JSON output: {res.stdout[:300]}") from e


def call_claude_stage1(prompt: str, model: str) -> dict:
    """Stage 1: return the structured {typed, committed} object."""
    for attempt in range(3):
        try:
            env = _run_claude(prompt, model, json_schema=STAGE1_SCHEMA)
            if env.get("is_error"):
                raise RuntimeError(f"claude error: {env.get('result', '')[:200]}")
            out = env.get("structured_output")
            if not out:
                # Fallback: try to parse the result field as JSON
                txt = env.get("result", "").strip()
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
                out = json.loads(cleaned)
            return out
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"  retry stage1 {attempt+1} after {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)


def call_claude_stage2(prompt: str, model: str) -> str:
    """Stage 2: return plain-text result (single word)."""
    for attempt in range(3):
        try:
            env = _run_claude(prompt, model)
            if env.get("is_error"):
                raise RuntimeError(f"claude error: {env.get('result', '')[:200]}")
            return (env.get("result") or "").strip()
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"  retry stage2 {attempt+1} after {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)


# ---------------------------- Seed sentence sampling ----------------------------

def load_seeds(seed_corpus: str, n: int, rng: random.Random) -> list[str]:
    """Sample n clean pt-BR sentences from a corpus shard."""
    lines = []
    with open(seed_corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Filter: medium-length, has at least one accent/cedilla (more interesting)
            if 30 <= len(line) <= 200 and re.search(r"[áéíóúâêîôûãõàèìòùç]", line.lower()):
                lines.append(line)
            if len(lines) >= n * 3:  # oversample, will subsample
                break
    rng.shuffle(lines)
    return lines[:n]


# ---------------------------- Main loop ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-corpus", required=True, help="Path to a clean pt-BR shard file (one sentence per line)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--target-pairs", type=int, default=750)
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Claude model. Haiku is plenty for typo generation and ~3x faster.")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel `claude -p` invocations.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sample-only", type=int, default=0,
                    help="If > 0, just emit N samples to stdout for eyeball check (no full run)")
    ap.add_argument("--priors-json", default=None,
                    help="Optional JSON file overriding CATEGORY_PRIORS (same keys). "
                         "Use for weak-spot-biased pools, e.g. prefix_completion=0.40, hybrid_multi=0.40.")
    args = ap.parse_args()

    if args.priors_json:
        override = json.loads(Path(args.priors_json).read_text())
        unknown = set(override) - set(CATEGORY_PRIORS)
        if unknown:
            sys.exit(f"--priors-json has unknown keys: {unknown}")
        CATEGORY_PRIORS.clear()
        CATEGORY_PRIORS.update(override)
        print(f"Using overridden priors: {CATEGORY_PRIORS}", file=sys.stderr)

    rng = random.Random(args.seed)
    if not shutil.which("claude"):
        sys.exit("`claude` CLI not on PATH. Install Claude Code or update PATH.")

    # Oversample seeds: many will be filtered out by round-trip verification
    n_seeds = args.target_pairs * 2 if args.sample_only == 0 else args.sample_only
    seeds = load_seeds(args.seed_corpus, n_seeds, rng)
    print(f"Loaded {len(seeds)} seed sentences", file=sys.stderr)

    accepted: list[dict] = []
    rejected_count = 0
    api_calls = 0
    state_lock = threading.Lock()
    stop_event = threading.Event()

    # Pre-build a weighted picker for the per-call category bias
    cat_keys = list(CATEGORY_PRIORS.keys())
    cat_weights = [CATEGORY_PRIORS[k] for k in cat_keys]

    def process_seed(i: int, sentence: str, chosen_cat: str) -> tuple[str, dict | None, str]:
        """Returns (status, pair_or_none, debug_msg). Status: 'ok' | 'reject' | 'err'."""
        category_hint = CATEGORY_HINTS.get(chosen_cat, "")
        try:
            obj = call_claude_stage1(
                STAGE1_PROMPT.format(sentence=sentence, category_hint=category_hint),
                args.model,
            )
            try:
                typed = obj["typed"].strip()
                committed = obj["committed"].strip()
            except (KeyError, AttributeError):
                return ("reject", None, f"bad stage1 object: {obj!r}")
            if not typed or not committed or typed == committed:
                return ("reject", None, "empty/identical")
            corrected = call_claude_stage2(STAGE2_PROMPT.format(typed=typed), args.model)
            corrected = corrected.strip().split()[0] if corrected.strip() else ""
            corrected = re.sub(r"^[\"']|[\"',.;:!?]$", "", corrected)
            if corrected.lower() != committed.lower():
                return ("reject", None, f"roundtrip {corrected!r} != {committed!r}")
            pair = {"typed": typed, "committed": committed,
                    "category_hint": chosen_cat,
                    "source": "claude_synth_v3", "seed_sentence": sentence}
            return ("ok", pair, "")
        except Exception as e:
            return ("err", None, f"{type(e).__name__}: {e}")

    # Pre-pick categories for each seed so we get deterministic distribution
    work_items = [(i, s, rng.choices(cat_keys, weights=cat_weights, k=1)[0])
                  for i, s in enumerate(seeds)]

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(process_seed, *w): w for w in work_items}
        try:
            for fut in as_completed(futures):
                if stop_event.is_set():
                    break
                i, sentence, _cat = futures[fut]
                with state_lock:
                    api_calls += 2  # always 2 calls per task (best-case)
                status, pair, msg = fut.result()
                with state_lock:
                    if status == "ok":
                        accepted.append(pair)
                        if len(accepted) % 25 == 0:
                            print(f"  accepted={len(accepted)} rejected={rejected_count} "
                                  f"api_calls={api_calls}", file=sys.stderr)
                        if len(accepted) >= args.target_pairs:
                            stop_event.set()
                    else:
                        rejected_count += 1
                        if status == "err":
                            print(f"  [{i}] error: {msg}", file=sys.stderr)
        except KeyboardInterrupt:
            stop_event.set()
            print("Interrupted, saving what we have...", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2))
    print(f"\nDONE: wrote {len(accepted)} pairs to {out_path}", file=sys.stderr)
    print(f"  rejected={rejected_count}  total_api_calls={api_calls}", file=sys.stderr)

    # Eyeball sample for stdout
    print("\n=== Sample of 10 accepted pairs ===", file=sys.stderr)
    for p in random.sample(accepted, min(10, len(accepted))):
        print(f"  {p['typed']!r:30s} -> {p['committed']!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
