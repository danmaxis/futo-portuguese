"""
Typo synthesis for the FUTO autocorrect format `<XBU>typo<XBC>correct<XEC>`.

Used by Phase 4a (isolated triples) and Phase 4b (in-context corrections).

Strategies (mixed per-word with weighted probabilities):
  1. Keyboard-adjacency typos using the pt-BR ABNT2 / pt-BR international
     layout (which is what Brazilians overwhelmingly use on mobile and desktop).
  2. Missing-accent / accent-confusion: drop or swap diacritics — the dominant
     pt-BR typo class. Examples: você → voce, está → esta, não → nao, café → cafe.
  3. Cedilla loss: ç → c (e.g. coração → coracao).
  4. Transposed adjacent letters (cab → bac).
  5. Single-char insertion (random letter from a likely class).
  6. Single-char deletion.
  7. Doubled char (e.g. comer → commer).
  8. Common pt_BR shortcut substitutions (vc, tb, hj, q, ñ, mt, etc.) —
     these are *known* corrections, weighted higher than synthetic typos.

The synthesis is deterministic given a seed so we can regenerate the same
training set, and respects a `log(word_freq + 1)` dampener so common words
don't flood the dataset.
"""
from __future__ import annotations
import math
import random
import re
import unicodedata
from typing import Callable

# pt-BR ABNT2 / "Português (Brasil)" layout adjacency map for the alpha rows.
# Each key maps to its left/right/up/down neighbours on the QWERTY-derived layout
# Brazilians use. Punctuation/diacritic dead-keys are intentionally omitted —
# those are handled by the "missing accent" rule, which is more common anyway.
ADJ = {
    "q": "wa12", "w": "qe23sa", "e": "wr34ds", "r": "et45fd", "t": "ry56gf",
    "y": "tu67hg", "u": "yi78jh", "i": "uo89kj", "o": "ip90lk", "p": "o0-l",
    "a": "qsz",   "s": "awxd",   "d": "secxf",  "f": "drcvg",  "g": "ftvbh",
    "h": "gybnj", "j": "hubmnk", "k": "jimol",  "l": "kopç",
    "ç": "lop",   # ABNT2 has a dedicated ç key right of L
    "z": "asx",   "x": "zsdc",   "c": "xdvfg",  "v": "cfbg",   "b": "vghn",
    "n": "bhjmk", "m": "njkl",
}

# Brazilian shortcut-style pt_BR misspellings: token → list of plausible
# "wrong" forms a typist might write instead. Treated as a high-quality
# synthetic typo class because they map to single recognised corrections.
PTBR_SHORTCUTS: dict[str, list[str]] = {
    "você":      ["vc", "voce", "vcê", "voçe"],
    "vocês":     ["vcs", "voces", "vocês"],
    "também":    ["tb", "tbm", "tambem", "tambm", "também"],
    "porque":    ["pq", "porq", "pq.", "pqq"],
    "que":       ["q", "ki", "qe"],
    "muito":     ["mt", "mto", "muto", "muiito"],
    "muita":     ["mta", "muta", "muiita"],
    "estou":     ["tô", "to", "to'", "tow"],
    "está":      ["tá", "ta", "tah", "tava"],
    "estava":    ["tava", "tva", "estva"],
    "obrigado":  ["obg", "obgd", "obrigad", "obrigda", "obrigaado"],
    "obrigada":  ["obga", "obrigda", "obrigaada"],
    "valeu":     ["vlw", "vle", "valew", "vlew"],
    "beleza":    ["blz", "blza", "blz!"],
    "falou":     ["flw", "falo", "flou"],
    "abraço":    ["abc", "abrc", "abraco"],
    "abraços":   ["abs", "abracos", "abrcs"],
    "beijo":     ["bj", "bjo", "beijoo"],
    "beijos":    ["bjs", "beijoss", "bjss"],
    "hoje":      ["hj", "hjj", "hoje", "hje"],
    "ninguém":   ["ngm", "ninguem", "niguem"],
    "tudo":      ["td", "tdo", "tudoo"],
    "todos":     ["tds", "tdos", "todus"],
    "agora":     ["agr", "agra", "ago"],
    "depois":    ["dps", "dpois", "depo"],
    "amigo":     ["amg", "migo", "amigu"],
    "amigos":    ["amgs", "migos", "amigus"],
    "mesmo":     ["msm", "mesmuu", "msmo"],
    "também":    ["tb", "tbm"],
    "para":      ["pra", "pro", "p", "p/"],
    "está":      ["ta", "tá"],
    "não":       ["nao", "ñ", "n", "naum", "nãoo"],
    "é":         ["eh"],
    "café":      ["cafe", "kafe"],
    "coração":   ["coracao", "coraçao"],
}


def _strip_accents(s: str) -> str:
    """ã → a, é → e, ç → c. NFD-decompose then drop combining marks + ç special-case."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn":
            continue
        if ch == "ç":
            out.append("c")
        elif ch == "Ç":
            out.append("C")
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _adj_typo(w: str, rng: random.Random) -> str:
    if not w:
        return w
    chars = list(w)
    candidates = [i for i, c in enumerate(chars) if c.lower() in ADJ]
    if not candidates:
        return w
    i = rng.choice(candidates)
    c = chars[i].lower()
    neighbours = ADJ[c]
    if not neighbours:
        return w
    new_c = rng.choice(neighbours)
    if chars[i].isupper():
        new_c = new_c.upper()
    chars[i] = new_c
    return "".join(chars)


def _drop_accent(w: str, rng: random.Random) -> str:
    """Targeted accent drop: only fires if the word has at least one accented char."""
    has_accent = any(unicodedata.combining(ch) or ch in "ç" for ch in unicodedata.normalize("NFD", w))
    if not has_accent:
        return w
    return _strip_accents(w)


def _swap_accent(w: str, rng: random.Random) -> str:
    """Swap one accented vowel with a different accent of the same letter, e.g. é→ê, á→ã."""
    swaps = {"á": "â", "â": "á", "ã": "â", "é": "ê", "ê": "é", "í": "î", "ó": "ô", "ô": "ó", "õ": "ô", "ú": "û"}
    chars = list(w)
    candidates = [i for i, c in enumerate(chars) if c.lower() in swaps]
    if not candidates:
        return _drop_accent(w, rng)
    i = rng.choice(candidates)
    c = chars[i]
    swapped = swaps[c.lower()]
    chars[i] = swapped.upper() if c.isupper() else swapped
    return "".join(chars)


def _transpose(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i+1] + w[i] + w[i+2:]


def _insert(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w) + 1)
    extra = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return w[:i] + extra + w[i:]


def _delete(w: str, rng: random.Random) -> str:
    if len(w) <= 2:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i+1:]


def _double(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i] + w[i:]


def _shortcut(w: str, rng: random.Random) -> str:
    """If we have a known shortcut for this word, return one of the shortcut forms."""
    forms = PTBR_SHORTCUTS.get(w.lower())
    if not forms:
        return None  # caller should fall back to another rule
    pick = rng.choice(forms)
    if w[0:1].isupper() and pick:
        pick = pick[0].upper() + pick[1:]
    return pick


# Per-rule weight: chosen to overweight realistic pt_BR typo classes
RULES: list[tuple[Callable, int]] = [
    (_drop_accent, 35),    # the dominant pt_BR typo class
    (_swap_accent, 10),
    (_adj_typo,    20),
    (_transpose,   10),
    (_delete,       8),
    (_insert,       7),
    (_double,       5),
]


def synth_typo(word: str, rng: random.Random) -> str | None:
    """Generate one plausible typo for `word`. Returns None if word is too short or noise.

    Strategy:
      - Try a known shortcut (high quality) ~25% of the time, if available.
      - Otherwise pick a synthetic rule by weight.
      - If the synthetic rule produces the original (no-op), retry once.
    """
    if len(word) < 2 or not re.match(r"^[A-Za-zÀ-ÿ'-]+$", word):
        return None
    # 25% known-shortcut bias
    if rng.random() < 0.25:
        s = _shortcut(word, rng)
        if s is not None and s != word:
            return s
    # Synthetic rule
    rules, weights = zip(*RULES)
    for _ in range(2):
        rule = rng.choices(rules, weights=weights, k=1)[0]
        out = rule(word, rng)
        if out != word:
            return out
    return None


def freq_weight_log(freq: int) -> float:
    """log(freq + 1) frequency dampener (per FUTO wiki recommendation)."""
    return math.log(freq + 1)


def to_keypress_chars(typed: str) -> list[str]:
    """Convert a string to <CHAR_X> tokens (ASCII A-Z only).
    Strips accents (ã→A, ç→C, é→E) and case. Non-letter chars dropped.
    This is the FUTO Keyboard's keypress-token format — what the model is
    actually trained on (verified via reference English model inference).
    """
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


def make_xbu_triple(typo: str, correct: str) -> str:
    """FUTO autocorrect format: <XBU><CHAR_*>...<CHAR_*><XBC>correct<XEC>.
    The TYPED part is keypresses (one <CHAR_X> per stroke); the CORRECTION
    part is the actual word in plain text. Verified against the reference
    English model via inference (see eval_english_baseline.py)."""
    chars = "".join(to_keypress_chars(typo))
    return f"<XBU>{chars}<XBC>{correct}<XEC>"


def make_inline_corrected(text: str, rng: random.Random, typo_rate: float = 0.33) -> str:
    """Phase 4b format: take a sentence, randomly replace ~typo_rate of words
    with <XBU><CHAR_*>...<CHAR_*><XBC>correct<XEC>. Words too short / non-alphabetic are skipped."""
    out_words: list[str] = []
    for word in text.split():
        # Strip leading/trailing punctuation; keep it around the corrected form.
        m = re.match(r"^([^A-Za-zÀ-ÿ']*)([A-Za-zÀ-ÿ']+)([^A-Za-zÀ-ÿ']*)$", word)
        if not m or rng.random() > typo_rate:
            out_words.append(word)
            continue
        prefix, core, suffix = m.groups()
        typo = synth_typo(core, rng)
        if typo is None:
            out_words.append(word)
            continue
        out_words.append(prefix + make_xbu_triple(typo, core) + suffix)
    return " ".join(out_words)


# ---------------------------- Self-test ----------------------------

def _selftest() -> None:
    rng = random.Random(42)
    samples = [
        "Bom dia, como você está hoje?",
        "Eu fui ao mercado comprar pão e leite.",
        "Não esqueci do nosso encontro amanhã na padaria.",
        "Obrigado pela ajuda com o projeto, valeu mesmo.",
        "Hoje tem futebol no estádio do Maracanã.",
    ]
    print("=== Phase 4b inline format ===")
    for s in samples:
        out = make_inline_corrected(s, rng, typo_rate=0.4)
        print(f"  {out}")
    print()
    print("=== Phase 4a isolated triples (sample 30) ===")
    for word in ["você", "também", "obrigado", "mercado", "amanhã", "padaria",
                 "coração", "estação", "café", "música", "número", "rapidamente",
                 "computador", "celular", "telefone", "trabalho", "dinheiro",
                 "domingo", "segunda", "exemplo", "natural", "verdade",
                 "português", "brasileiro", "amigo", "família", "história",
                 "questão", "lição", "ação"]:
        typo = synth_typo(word, rng)
        if typo is None:
            print(f"  [skip] {word}")
            continue
        print(f"  {make_xbu_triple(typo, word)}")


if __name__ == "__main__":
    _selftest()
