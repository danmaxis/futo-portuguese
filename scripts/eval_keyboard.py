"""
Evaluation harness for the pt_BR FUTO keyboard model.

Two test suites:
  1. **autocorrect**: feed `<XBU>typo<XBC>` and check that the model
     completes with the correct word followed by `<XEC>`. Top-1 + top-5 acc.
  2. **next-word**: feed a pt_BR sentence prefix and check that the next-token
     prediction is reasonable (top-1 in a small target set).

Usage:
  # On the GPU host, after Phase 3 pretrain:
  python eval_keyboard.py --checkpoint pretrain/base --tokenizer tokenizer/spm_pt_br.model

  # After Phase 4 fine-tune:
  python eval_keyboard.py --checkpoint finetune/stage_b/final --tokenizer tokenizer/spm_pt_br.model
"""
from __future__ import annotations
import argparse
import json
import unicodedata
from pathlib import Path

import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM


def to_keypress_chars(typed: str) -> list[str]:
    """Convert a string to <CHAR_X> tokens (ASCII A-Z only).
    Strips accents (ã→A, ç→C, é→E) and case. Mirrors lib_typo_synthesis."""
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


# Hand-curated pt_BR autocorrect test set (typo, expected_correction)
AUTOCORRECT_TESTS = [
    # Common shortcuts
    ("vc", "você"),
    ("tb", "também"),
    ("hj", "hoje"),
    ("pq", "porque"),
    ("obg", "obrigado"),
    ("vlw", "valeu"),
    ("blz", "beleza"),
    ("mt", "muito"),
    # Missing accents (the dominant pt_BR typo class)
    ("voce", "você"),
    ("tambem", "também"),
    ("nao", "não"),
    ("esta", "está"),
    ("portugues", "português"),
    ("estao", "estão"),
    ("orgao", "órgão"),
    ("avo", "avó"),
    # Cedilla loss
    ("coracao", "coração"),
    ("acao", "ação"),
    ("relacao", "relação"),
    ("estacao", "estação"),
    # Adjacency typos
    ("merrcado", "mercado"),
    ("tdabalho", "trabalho"),
    ("compitador", "computador"),
    # Transposed
    ("aamnha", "amanhã"),
    ("cidaade", "cidade"),
    # Doubled
    ("commer", "comer"),
    ("livvro", "livro"),
    # Common pt_BR misspellings
    ("execelente", "excelente"),
    ("rezolver", "resolver"),
    ("agradeso", "agradeço"),
]


# (prefix, plausible-next-words) for next-word evaluation
NEXT_WORD_TESTS = [
    ("Bom dia, como você", ["está", "vai", "passou", "tem"]),
    ("Os meninos foram para a", ["escola", "casa", "praia", "rua"]),
    ("Eu queria um café com", ["leite", "açúcar", "creme", "bolo"]),
    ("A capital do Brasil é", ["Brasília"]),
    ("Hoje está fazendo muito", ["calor", "frio", "sol", "vento"]),
    ("Eu não estou", ["bem", "feliz", "indo", "conseguindo", "sabendo", "entendendo"]),
    ("O jogo do Brasil contra a", ["Argentina", "Alemanha", "França", "Itália", "Espanha"]),
]


def encode_xbu(sp, typo):
    """Encode <XBU><CHAR_*>...<CHAR_*><XBC> using FUTO keypress format.
    Each char of `typo` becomes a <CHAR_X> token (accent-stripped, uppercase)."""
    ids = [sp.piece_to_id("<XBU>")]
    for piece in to_keypress_chars(typo):
        ids.append(sp.piece_to_id(piece))
    ids.append(sp.piece_to_id("<XBC>"))
    return ids


def decode_until_xec(sp, ids):
    """Decode token IDs back to text, stopping at <XEC> if present."""
    xec = sp.piece_to_id("<XEC>")
    if xec in ids:
        ids = ids[: ids.index(xec)]
    return sp.decode(ids)


def evaluate_autocorrect(model, sp, device, tests, max_new_tokens=12, top_k=5):
    """For each (typo, correct), feed <XBU>typo<XBC> and greedy-decode up to <XEC>."""
    xec = sp.piece_to_id("<XEC>")
    pad = sp.pad_id()

    top1_correct = 0
    top5_correct = 0
    rows = []

    for typo, correct in tests:
        prompt_ids = encode_xbu(sp, typo)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Greedy decode
        with torch.no_grad():
            out = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad,
                eos_token_id=xec,
            )
        gen_ids = out[0, len(prompt_ids):].tolist()
        prediction = decode_until_xec(sp, gen_ids).strip()

        # Top-5 via beam search
        with torch.no_grad():
            beams = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                num_beams=top_k,
                num_return_sequences=top_k,
                do_sample=False,
                pad_token_id=pad,
                eos_token_id=xec,
                early_stopping=True,
            )
        top5_predictions = []
        for k in range(beams.shape[0]):
            ids = beams[k, len(prompt_ids):].tolist()
            top5_predictions.append(decode_until_xec(sp, ids).strip())

        is_top1 = prediction.lower() == correct.lower()
        is_top5 = correct.lower() in {p.lower() for p in top5_predictions}
        if is_top1: top1_correct += 1
        if is_top5: top5_correct += 1
        rows.append({
            "typo": typo, "correct": correct, "top1": prediction,
            "top5": top5_predictions, "top1_hit": is_top1, "top5_hit": is_top5,
        })

    return top1_correct, top5_correct, rows


def evaluate_next_word(model, sp, device, tests, max_new_tokens=4, top_k=8):
    """For each (prefix, plausible_words), check whether top-1 / top-k matches any plausible word."""
    pad = sp.pad_id()

    top1_correct = 0
    topk_correct = 0
    rows = []

    for prefix, plausible in tests:
        prompt_ids = sp.encode(prefix, out_type=int)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad,
            )
        gen_ids = out[0, len(prompt_ids):].tolist()
        # Decode and take the first token-word
        gen_text = sp.decode(gen_ids)
        first_word = gen_text.strip().split()[0] if gen_text.strip() else ""

        with torch.no_grad():
            beams = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                num_beams=top_k,
                num_return_sequences=top_k,
                do_sample=False,
                pad_token_id=pad,
                early_stopping=True,
            )
        topk_first_words = set()
        for k in range(beams.shape[0]):
            ids = beams[k, len(prompt_ids):].tolist()
            txt = sp.decode(ids).strip()
            if txt:
                topk_first_words.add(txt.split()[0].lower().strip(",.;:!?"))

        plausible_lower = {p.lower() for p in plausible}
        is_top1 = first_word.lower().strip(",.;:!?") in plausible_lower
        is_topk = bool(topk_first_words & plausible_lower)
        if is_top1: top1_correct += 1
        if is_topk: topk_correct += 1
        rows.append({
            "prefix": prefix, "plausible": plausible, "top1": first_word,
            "topk": sorted(topk_first_words), "top1_hit": is_top1, "topk_hit": is_topk,
        })

    return top1_correct, topk_correct, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="notes/eval_results.json")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Loading {args.checkpoint} on {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"  tokenizer vocab: {sp.get_piece_size()}")

    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.eval().to(device)
    print(f"  model loaded, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    print("\n=== Autocorrect (XBU/XBC/XEC) ===")
    ac_top1, ac_top5, ac_rows = evaluate_autocorrect(model, sp, device, AUTOCORRECT_TESTS)
    n = len(AUTOCORRECT_TESTS)
    print(f"  top-1: {ac_top1}/{n} = {100*ac_top1/n:.1f}%")
    print(f"  top-5: {ac_top5}/{n} = {100*ac_top5/n:.1f}%")
    print("  details:")
    for r in ac_rows:
        flag = "✓" if r["top1_hit"] else ("◯" if r["top5_hit"] else "✗")
        print(f"    {flag} {r['typo']!r:<14} → {r['top1']!r:<14} (want {r['correct']!r})")

    print("\n=== Next-word (free continuation) ===")
    nw_top1, nw_topk, nw_rows = evaluate_next_word(model, sp, device, NEXT_WORD_TESTS)
    n = len(NEXT_WORD_TESTS)
    print(f"  top-1: {nw_top1}/{n} = {100*nw_top1/n:.1f}%")
    print(f"  top-8: {nw_topk}/{n} = {100*nw_topk/n:.1f}%")
    print("  details:")
    for r in nw_rows:
        flag = "✓" if r["top1_hit"] else ("◯" if r["topk_hit"] else "✗")
        print(f"    {flag} {r['prefix']!r}")
        print(f"        top1: {r['top1']!r}; topk: {r['topk']}")

    # JSON dump
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "autocorrect": {
            "top1": ac_top1, "top5": ac_top5, "n": len(AUTOCORRECT_TESTS),
            "rows": ac_rows,
        },
        "next_word": {
            "top1": nw_top1, "topk": nw_topk, "n": len(NEXT_WORD_TESTS),
            "rows": nw_rows,
        },
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
