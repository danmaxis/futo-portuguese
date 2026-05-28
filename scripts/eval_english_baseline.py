"""
Baseline evaluation: run the official English FUTO model on autocorrect tests,
to validate our methodology and serve as a reference quality bar for pt_BR.

Architectural note: the FUTO Android app reads its tokenizer from the embedded
`keyboardlm.ext_tokenizer_data` SentencePiece bytes, NOT from the GGUF's
`tokenizer.ggml.*` fields. We mirror that here — load the extracted SP for
encoding/decoding, and use llama_cpp only as a weight executor.

Uses llama-cpp-python (CPU is fine for the 36M model — ~10-50 ms/token).

Usage:
  python eval_english_baseline.py
  python eval_english_baseline.py --model reference_model/ml4_1_f16_meta_fixed.gguf
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import sentencepiece as spm
from llama_cpp import Llama


# English autocorrect test set — mirrors the structure of our pt_BR tests.
# Each entry: (typo, expected_correction, category)
ENGLISH_AUTOCORRECT = [
    # Common shortcuts / contractions
    ("youre",       "you're",     "shortcut"),
    ("im",          "I'm",        "shortcut"),
    ("dont",        "don't",      "shortcut"),
    ("cant",        "can't",      "shortcut"),
    ("wont",        "won't",      "shortcut"),
    ("isnt",        "isn't",      "shortcut"),
    ("its",         "it's",       "shortcut"),  # ambiguous — "its" is also valid
    ("hes",         "he's",       "shortcut"),
    # Adjacency typos (qwerty)
    ("teh",         "the",        "adjacency"),
    ("recieve",     "receive",    "adjacency"),
    ("seperate",    "separate",   "adjacency"),
    ("definately",  "definitely", "adjacency"),
    ("hapen",       "happen",     "adjacency"),
    ("merrcado",    "mercado",    "adjacency"),  # not English; should fail or fallback
    # Transposed
    ("freind",      "friend",     "transpose"),
    ("wierd",       "weird",      "transpose"),
    ("becuase",     "because",    "transpose"),
    # Doubled
    ("commitee",    "committee",  "doubled"),  # actually missing a 't' but realistic
    ("acommodate",  "accommodate","doubled"),
    # Common misspellings
    ("alot",        "a lot",      "misspell"),  # ambiguous — model may complete differently
    ("untill",      "until",      "misspell"),
    ("occured",     "occurred",   "misspell"),
    ("tomorow",     "tomorrow",   "misspell"),
    ("begining",    "beginning",  "misspell"),
    # Easy ones — model should crush these
    ("hte",         "the",        "trivial"),
    ("adn",         "and",        "trivial"),
    ("yuo",         "you",        "trivial"),
]


# English next-word tests (analogous to our pt_BR set)
ENGLISH_NEXT_WORD = [
    ("Good morning, how are",   ["you"]),
    ("The boys went to the",    ["store", "park", "school", "beach", "house", "game"]),
    ("I would like a coffee with", ["milk", "sugar", "cream"]),
    ("The capital of France is", ["Paris"]),
    ("It is very",              ["hot", "cold", "nice", "good", "bad", "important"]),
    ("I'm not feeling",         ["well", "good", "great", "right"]),
    ("She said she would",      ["come", "go", "be", "do", "meet", "see", "leave", "try"]),
]


import unicodedata


def to_keypress_chars(text: str) -> list[str]:
    """Convert a typed string to <CHAR_X> tokens (ASCII A-Z only).
    Strips accents (ã→A, ç→C, é→E) and case (case-insensitive on a keyboard)."""
    out = []
    for ch in text:
        # Decompose: ã → a + COMBINING TILDE; drop the combining marks
        decomposed = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in decomposed if not unicodedata.combining(c))
        if base == "ç" or base == "Ç":
            base = "C"
        upper = base.upper()
        for c in upper:
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out


def encode_xbu(sp: spm.SentencePieceProcessor, typo: str) -> list[int]:
    """Encode <XBU><CHAR_*>...<CHAR_*><XBC> as token IDs (FUTO keypress format).
    Each character of `typo` becomes a <CHAR_X> token. Non-alphabetic chars dropped.
    """
    ids = [sp.piece_to_id("<XBU>")]
    for piece in to_keypress_chars(typo):
        ids.append(sp.piece_to_id(piece))
    ids.append(sp.piece_to_id("<XBC>"))
    return ids


def greedy_complete(llm: Llama, prompt_ids: list[int], max_tokens: int,
                    stop_ids: set[int]) -> list[int]:
    """Greedy decode until stop_id appears or max_tokens reached."""
    out: list[int] = []
    llm.reset()
    llm.eval(prompt_ids)
    for _ in range(max_tokens):
        logits = llm.scores[llm.n_tokens - 1]
        next_id = int(logits.argmax())
        if next_id in stop_ids:
            break
        out.append(next_id)
        llm.eval([next_id])
    return out


def topk_complete(llm: Llama, prompt_ids: list[int], max_tokens: int,
                  stop_ids: set[int], k: int = 5):
    """Sample k continuations at temp 0.6 — proxy for top-k beams."""
    candidates: set[tuple] = set()
    for _ in range(k * 4):
        if len(candidates) >= k:
            break
        llm.reset()
        llm.eval(prompt_ids)
        out: list[int] = []
        for _ in range(max_tokens):
            tok = llm.sample(temp=0.6, top_k=20, top_p=0.95)
            if tok in stop_ids:
                break
            out.append(tok)
            llm.eval([tok])
        if out:
            candidates.add(tuple(out))
    return [list(c) for c in candidates]


def first_word(text: str) -> str:
    return text.strip().split()[0].lower().strip(",.;:!?'\"") if text.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reference_model/ml4_1_f16_meta_fixed.gguf")
    ap.add_argument("--tokenizer", default="reference_model/extracted_spm.model",
                    help="SentencePiece model file (extracted from GGUF's keyboardlm.ext_tokenizer_data)")
    ap.add_argument("--out", default="notes/eval_english_baseline.json")
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=12)
    args = ap.parse_args()

    print(f"Loading SentencePiece: {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"  vocab_size = {sp.get_piece_size()}")

    # Verify the slot map matches the reference (XBU/XBC/XEC at 174/175/176)
    xbu_id = sp.piece_to_id("<XBU>")
    xbc_id = sp.piece_to_id("<XBC>")
    xec_id = sp.piece_to_id("<XEC>")
    char_a_id = sp.piece_to_id("<CHAR_A>")
    print(f"  <XBU>={xbu_id} <XBC>={xbc_id} <XEC>={xec_id} <CHAR_A>={char_a_id}")
    assert xbu_id == 174 and xbc_id == 175 and xec_id == 176, "Slot map mismatch — schema changed?"
    stop_ids: set[int] = {xec_id, sp.eos_id()}

    print(f"\nLoading model: {args.model}")
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=4, verbose=False,
                logits_all=True)
    print()

    # ---------------- Autocorrect ----------------
    print("=== English autocorrect (XBU/XBC/XEC) ===")
    n = len(ENGLISH_AUTOCORRECT)
    top1_correct = 0
    top5_correct = 0
    by_category: dict = {}
    rows = []
    t0 = time.time()
    for typo, correct, category in ENGLISH_AUTOCORRECT:
        prompt = encode_xbu(sp, typo)
        gen_ids = greedy_complete(llm, prompt, args.max_tokens, stop_ids)
        prediction = sp.decode(gen_ids).strip()
        topk_id_lists = topk_complete(llm, prompt, args.max_tokens, stop_ids, k=5)
        topk = [sp.decode(ids).strip() for ids in topk_id_lists]

        is_top1 = prediction.lower() == correct.lower()
        is_top5 = correct.lower() in {p.lower() for p in topk}
        if is_top1: top1_correct += 1
        if is_top5: top5_correct += 1

        cat = by_category.setdefault(category, {"top1": 0, "top5": 0, "n": 0})
        cat["n"] += 1
        if is_top1: cat["top1"] += 1
        if is_top5: cat["top5"] += 1

        flag = "✓" if is_top1 else ("◯" if is_top5 else "✗")
        print(f"  {flag} [{category:9}] {typo!r:<14} → {prediction!r:<14} (want {correct!r})")
        rows.append({"typo": typo, "correct": correct, "category": category,
                     "top1": prediction, "top5": topk,
                     "top1_hit": is_top1, "top5_hit": is_top5})
    elapsed = time.time() - t0
    print()
    print(f"  top-1: {top1_correct}/{n} = {100*top1_correct/n:.1f}%")
    print(f"  top-5: {top5_correct}/{n} = {100*top5_correct/n:.1f}%")
    print(f"  By category:")
    for cat, v in sorted(by_category.items()):
        print(f"    {cat:10}: top1={v['top1']}/{v['n']} ({100*v['top1']/v['n']:.0f}%) top5={v['top5']}/{v['n']} ({100*v['top5']/v['n']:.0f}%)")
    print(f"  ({elapsed:.1f}s, {elapsed/n*1000:.0f} ms/test)")

    # ---------------- Next-word ----------------
    print()
    print("=== English next-word ===")
    n2 = len(ENGLISH_NEXT_WORD)
    nw_top1 = 0
    nw_topk = 0
    nw_rows = []
    for prefix, plausible in ENGLISH_NEXT_WORD:
        ids = [sp.bos_id()] + sp.encode(prefix, out_type=int)
        gen = greedy_complete(llm, ids, 5, {sp.eos_id()})
        first_text = sp.decode(gen)
        fw = first_word(first_text)

        topk_id_lists = topk_complete(llm, ids, 5, {sp.eos_id()}, k=8)
        topk_words = {first_word(sp.decode(ids)) for ids in topk_id_lists}
        topk_words.discard("")

        plausible_lower = {p.lower() for p in plausible}
        is_top1 = fw in plausible_lower
        is_topk = bool(topk_words & plausible_lower)
        if is_top1: nw_top1 += 1
        if is_topk: nw_topk += 1
        flag = "✓" if is_top1 else ("◯" if is_topk else "✗")
        print(f"  {flag} {prefix!r}")
        print(f"      top1: {fw!r}; topk: {sorted(topk_words)}")
        nw_rows.append({"prefix": prefix, "plausible": plausible,
                        "top1": fw, "topk": sorted(topk_words),
                        "top1_hit": is_top1, "topk_hit": is_topk})
    print()
    print(f"  next-word top-1: {nw_top1}/{n2} = {100*nw_top1/n2:.1f}%")
    print(f"  next-word top-8: {nw_topk}/{n2} = {100*nw_topk/n2:.1f}%")

    # JSON dump
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps({
        "model": args.model,
        "autocorrect": {
            "n": n, "top1": top1_correct, "top5": top5_correct,
            "by_category": by_category, "rows": rows,
        },
        "next_word": {
            "n": n2, "top1": nw_top1, "topk": nw_topk, "rows": nw_rows,
        },
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
