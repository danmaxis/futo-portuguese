"""
Process the user's collected typo_log.jsonl into:
  - notes/real_typos_eval.json   — 50 hold-out pairs, NEVER used in training
  - notes/real_typos_pool.json   — remaining real pairs for training mix
  - notes/synth_typos.json       — large synthetic pool matching the real distribution
  - notes/real_typo_stats.json   — summary stats + per-category samples

Distribution is measured from the real log, then synth is generated to mirror it.
Outputs feed Phase 4a/4b retrain with --real-mix-ratio.

Usage:
  python scripts/07_process_typo_log.py \
      --log notes/typo_log.jsonl \
      --wordfreq notes/wordfreq.json \
      --synth-count 50000
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


# ABNT2 Brazilian Portuguese keyboard layout, qwerty base with ç at the right of L.
# Each key maps to the row neighbours likely to be hit by mistake. Only alpha keys.
ABNT2 = {
    'q': 'wa',     'w': 'qeas',   'e': 'wrsd',   'r': 'etdf',   't': 'ryfg',
    'y': 'tugh',   'u': 'yihj',   'i': 'uojk',   'o': 'ipkl',   'p': 'olç',
    'a': 'qwsz',   's': 'awedxz', 'd': 'erfcx',  'f': 'rtgvc',  'g': 'tyhbv',
    'h': 'yujnb',  'j': 'uikmn',  'k': 'iolm',   'l': 'opç',    'ç': 'lp',
    'z': 'asx',    'x': 'sdcz',   'c': 'dfvx',   'v': 'fgbc',   'b': 'ghnv',
    'n': 'hjmb',   'm': 'jkn',
}


# ---------------------------------------------------------------------------
# Real-typo categorisation (matches our analysis pipeline)
# ---------------------------------------------------------------------------

def strip_accents(s: str) -> str:
    out = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.combining(c):
            continue
        if c == 'ç':
            out.append('c')
        elif c == 'Ç':
            out.append('C')
        else:
            out.append(c)
    return ''.join(out)


def categorize(t: str, c: str) -> str:
    if t == c:
        return 'identical'
    t_lo, c_lo = t.lower(), c.lower()
    t_strip = strip_accents(t).lower()
    c_strip = strip_accents(c).lower()
    if t_lo == c_lo:
        return 'capitalization'
    if t_strip == c_strip and 'ç' in c.lower() and 'ç' not in t.lower():
        return 'cedilla_only'
    if t_strip == c_strip:
        return 'accent_only'
    if ' ' in c and ' ' not in t and strip_accents(c.replace(' ', '')).lower() == t_strip:
        return 'mwe_split'
    if c_lo.startswith(t_lo) and len(c) > len(t):
        return 'prefix_completion'
    if abs(len(t) - len(c)) <= 1:
        common = sum(1 for a, b in zip(t_lo, c_lo) if a == b)
        if common >= max(len(t), len(c)) - 2:
            return 'adjacency_typo'
    return 'other'


# ---------------------------------------------------------------------------
# Synthetic generators — one per category
# ---------------------------------------------------------------------------

def gen_accent_only(word: str, rng: random.Random) -> str | None:
    stripped = strip_accents(word)
    return stripped if stripped != word else None


def gen_cedilla_only(word: str, rng: random.Random) -> str | None:
    if 'ç' not in word.lower():
        return None
    return word.replace('ç', 'c').replace('Ç', 'C')


# Small curated list of typically-capitalized pt-BR strings.
# wordfreq.json is lowercased, so we need a separate source for capitalization
# pairs. Real-log data shows people typing 'claude'->'Claude', 'app'->'APP', etc.
CAPITALIZED_WORDS = [
    # Proper nouns — names
    "João", "Maria", "Paulo", "Pedro", "Ana", "Lucas", "Carla", "Ricardo", "Bruno",
    "Renata", "Carolina", "Fernanda", "Mateus", "Gabriela", "Rafael", "Beatriz",
    "Felipe", "Camila", "Diego", "Júlia", "Daniel", "Letícia", "Gustavo", "Larissa",
    # Places
    "Brasil", "Brasília", "Bahia", "Paraná", "Pernambuco", "Goiás", "Ceará",
    "Rio", "Recife", "Salvador", "Curitiba", "Manaus", "Belém", "Fortaleza",
    "Portugal", "Lisboa", "Porto", "França", "Alemanha", "Itália", "Espanha",
    "Japão", "China", "Argentina", "Chile", "México", "Estados",
    # Brands / apps
    "Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix", "YouTube",
    "WhatsApp", "Instagram", "Twitter", "Spotify", "Uber", "iFood", "Mercado",
    "Claude", "ChatGPT", "OpenAI", "Anthropic", "GitHub", "GitLab",
    # Acronyms (UPPERCASE)
    "URL", "API", "HTTP", "HTTPS", "GPS", "USB", "OK", "TV", "PDF", "CPU", "GPU",
    "RAM", "SSD", "VPN", "IP", "DNS", "CSS", "HTML", "JSON", "SQL", "SMS",
    "DVD", "CD", "FM", "AM", "VHS", "DJ", "MC", "IBGE", "INSS", "CPF", "CNPJ",
    "UF", "RH", "PM", "AM", "OAB", "USP", "UFRJ", "PUC",
]


def gen_capitalization(word: str, rng: random.Random) -> str | None:
    """For capitalization, we IGNORE the wordfreq word (which is lowercased)
    and pick from a curated capitalized list. Generate (lowercase, capitalized) pair.
    """
    cap = rng.choice(CAPITALIZED_WORDS)
    return cap.lower()  # the "typed" version


def gen_prefix_completion(word: str, rng: random.Random) -> str | None:
    if len(word) < 3:
        return None
    # Real data: typed_len typically 30-70% of correct_len
    cut = rng.randint(2, max(2, int(len(word) * 0.7)))
    return word[:cut]


def gen_adjacency(word: str, rng: random.Random) -> str | None:
    if len(word) < 2:
        return None
    chars = list(word)
    pos = rng.randrange(len(chars))
    base = strip_accents(chars[pos]).lower()
    if base not in ABNT2:
        # fallback to another rule
        return gen_transpose(word, rng)
    new = rng.choice(ABNT2[base])
    if chars[pos].isupper():
        new = new.upper()
    chars[pos] = new
    return ''.join(chars)


def gen_transpose(word: str, rng: random.Random) -> str | None:
    if len(word) < 3:
        return None
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def gen_drop(word: str, rng: random.Random) -> str | None:
    if len(word) < 3:
        return None
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1:]


def gen_insert(word: str, rng: random.Random) -> str | None:
    if len(word) < 2:
        return None
    pos = rng.randrange(len(word) + 1)
    extra = rng.choice('abcdefghijklmnopqrstuvwxyz')
    return word[:pos] + extra + word[pos:]


def gen_hybrid(word: str, rng: random.Random) -> str | None:
    """Compose two simpler mechanics. Picks 2 random non-hybrid generators
    and applies them sequentially. ~12% of real typos have multi-class noise."""
    candidates = [gen_accent_only, gen_cedilla_only, gen_adjacency, gen_drop,
                  gen_transpose, gen_insert, gen_prefix_completion]
    g1 = rng.choice(candidates)
    g2 = rng.choice(candidates)
    out = g1(word, rng)
    if not out:
        return None
    out2 = g2(out, rng)
    if not out2 or out2 == word:
        return out  # one mechanic is fine if second failed
    return out2


GENERATORS = {
    'accent_only':       gen_accent_only,
    'cedilla_only':      gen_cedilla_only,
    'capitalization':    gen_capitalization,
    'prefix_completion': gen_prefix_completion,
    'adjacency_typo':    gen_adjacency,
    'hybrid':            gen_hybrid,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_word_list(wordfreq_path: Path, min_count: int = 50) -> list[tuple[str, int]]:
    """Load (word, count) pairs from wordfreq.json. Filter to alpha-only words ≥ min_count."""
    data = json.loads(wordfreq_path.read_text())
    out = []
    for w, n in data.items():
        if n < min_count:
            continue
        if not re.match(r'^[a-zA-ZÀ-ÿ]+$', w):
            continue
        if len(w) < 2:
            continue
        out.append((w, n))
    return out


def precompute_weighted_sampler(words: list[tuple[str, int]]):
    """Build a fast sampler using cumulative log-weights + bisect.
    O(log N) per sample instead of O(N) of random.choices."""
    import math
    import bisect
    word_list = [w for w, _ in words]
    weights = [math.log(n + 1) for _, n in words]
    cum = []
    total = 0.0
    for w in weights:
        total += w
        cum.append(total)
    def sample(rng: random.Random) -> str:
        x = rng.random() * total
        idx = bisect.bisect_left(cum, x)
        if idx >= len(word_list):
            idx = len(word_list) - 1
        return word_list[idx]
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='notes/typo_log.jsonl')
    ap.add_argument('--wordfreq', default='notes/wordfreq.json',
                    help='Word frequency JSON (from scripts/04a_build_wordfreq.py)')
    ap.add_argument('--out-dir', default='notes')
    ap.add_argument('--synth-count', type=int, default=50000,
                    help='Number of synthetic pairs to generate')
    ap.add_argument('--eval-size', type=int, default=50,
                    help='How many real pairs to hold out for eval (untouched in training)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ---- 1. Load + categorize real typos ----
    real = []
    with open(args.log) as f:
        for line in f:
            d = json.loads(line)
            d['category'] = categorize(d['typed'], d['committed'])
            real.append(d)
    cats = Counter(d['category'] for d in real)
    n_real = len(real)
    print(f'Real typos: {n_real} pairs')
    for k, v in cats.most_common():
        print(f'  {k:22s} {v:4d}  ({100*v/n_real:5.1f}%)')

    # ---- 2. Hold out eval set, dedup, save pool ----
    rng.shuffle(real)
    seen = set()
    unique = []
    for d in real:
        key = (d['typed'].lower(), d['committed'].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    print(f'\nAfter dedup: {len(unique)} unique pairs (was {n_real})')

    eval_set = unique[:args.eval_size]
    pool_real = unique[args.eval_size:]

    (out_dir / 'real_typos_eval.json').write_text(json.dumps(eval_set, ensure_ascii=False, indent=2))
    (out_dir / 'real_typos_pool.json').write_text(json.dumps(pool_real, ensure_ascii=False, indent=2))
    print(f'Saved {args.eval_size} eval + {len(pool_real)} train-pool real pairs')

    # ---- 3. Compute target distribution (exclude 'identical', 'mwe_split', 'other') ----
    target_cats = ['adjacency_typo', 'accent_only', 'prefix_completion',
                   'hybrid', 'capitalization', 'cedilla_only']
    # Map 'other' bucket onto hybrid (since 'other' is often multi-class noise)
    cats_norm = Counter()
    for c, n in cats.items():
        if c == 'other':
            cats_norm['hybrid'] += n
        elif c == 'mwe_split':
            continue  # skip MWE for now
        elif c == 'identical':
            continue
        else:
            cats_norm[c] += n
    total_norm = sum(cats_norm.values())
    dist = {k: cats_norm[k] / total_norm for k in target_cats if cats_norm[k] > 0}
    print(f'\nNormalized target distribution for synth:')
    for k, p in sorted(dist.items(), key=lambda x: -x[1]):
        print(f'  {k:22s} {100*p:5.1f}%')

    # ---- 4. Generate synthetic pool ----
    words = load_word_list(Path(args.wordfreq))
    print(f'\nGenerating {args.synth_count} synth pairs from {len(words)} source words')

    # Per-category word eligibility filters: many generators only succeed on
    # specific word shapes (e.g. capitalization needs first-char uppercase).
    # Build a per-category subset and one sampler each so the category
    # distribution actually matches the target.
    def has_diacritic(w):
        return any(unicodedata.combining(c) or c in 'çÇ'
                   for c in unicodedata.normalize('NFD', w))
    eligibility = {
        'accent_only':       lambda w: has_diacritic(w),
        'cedilla_only':      lambda w: 'ç' in w.lower(),
        'capitalization':    lambda w: w[0].isupper() and not w.isupper(),
        'prefix_completion': lambda w: len(w) >= 3,
        'adjacency_typo':    lambda w: len(w) >= 2,
        'hybrid':            lambda w: len(w) >= 2,
    }
    cat_samplers = {}
    for cat, ok in eligibility.items():
        subset = [(w, n) for w, n in words if ok(w)]
        if not subset:
            print(f'  WARN: no eligible words for {cat}')
            continue
        cat_samplers[cat] = (precompute_weighted_sampler(subset), len(subset))
        print(f'  {cat:22s} eligible_words={len(subset)}')

    # Category picker matching observed distribution
    # Note: 'capitalization' uses CAPITALIZED_WORDS (not wordfreq) so it's not
    # in cat_samplers, but we still want it in the picker.
    cats_for_pick = [c for c in dist if c in cat_samplers or c == 'capitalization']
    weights_for_pick = [dist[c] for c in cats_for_pick]
    cum_cat = []
    total_c = 0.0
    for w in weights_for_pick:
        total_c += w
        cum_cat.append(total_c)
    import bisect
    def pick_cat() -> str:
        x = rng.random() * total_c
        idx = bisect.bisect_left(cum_cat, x)
        return cats_for_pick[min(idx, len(cats_for_pick) - 1)]

    synth = []
    attempts = 0
    while len(synth) < args.synth_count and attempts < args.synth_count * 10:
        attempts += 1
        cat = pick_cat()
        # Capitalization is special: don't use wordfreq, pull from CAPITALIZED_WORDS list
        if cat == 'capitalization':
            cap_word = rng.choice(CAPITALIZED_WORDS)
            typed = cap_word.lower()
            if typed == cap_word:
                continue  # word was all-lowercase already
            synth.append({'typed': typed, 'committed': cap_word, 'category': cat})
            continue
        # Normal categories use the wordfreq-based per-category sampler
        if cat not in cat_samplers:
            continue
        pick_word, _ = cat_samplers[cat]
        word = pick_word(rng)
        typed = GENERATORS[cat](word, rng)
        if not typed or typed == word or len(typed) < 1:
            continue
        synth.append({'typed': typed, 'committed': word, 'category': cat})

    (out_dir / 'synth_typos.json').write_text(json.dumps(synth, ensure_ascii=False))
    print(f'Wrote {len(synth)} synth pairs to {out_dir/"synth_typos.json"} '
          f'(attempts: {attempts})')

    # ---- 5. Sanity-check synth distribution matches target ----
    synth_cats = Counter(d['category'] for d in synth)
    print('\nSynth category distribution (should match target):')
    for k, p in sorted(dist.items(), key=lambda x: -x[1]):
        actual = synth_cats[k] / len(synth)
        print(f'  {k:22s} target={100*p:5.1f}%  actual={100*actual:5.1f}%')

    # ---- 6. Stats summary ----
    stats = {
        'real_total': n_real,
        'real_unique': len(unique),
        'eval_size': args.eval_size,
        'train_pool_real': len(pool_real),
        'synth_total': len(synth),
        'real_categories': dict(cats),
        'synth_categories': dict(synth_cats),
        'distribution': dist,
    }
    (out_dir / 'real_typo_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'\nStats written to {out_dir/"real_typo_stats.json"}')


if __name__ == '__main__':
    main()
