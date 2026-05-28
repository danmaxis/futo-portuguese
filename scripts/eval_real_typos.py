"""
Evaluate a checkpoint against the 50-pair real-user typo hold-out
(notes/real_typos_eval.json — never seen during training).

Each pair: {"typed": "voc", "committed": "você", ...}
Prompt: <XBU><CHAR_V><CHAR_O><CHAR_C><XBC>
Expected output: você<XEC>

Reports top-1, top-5 (greedy + beam search) and per-category breakdown
when category info is present.
"""
from __future__ import annotations
import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM


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


def encode_xbu(sp, typo):
    ids = [sp.piece_to_id("<XBU>")]
    for tok in to_keypress_chars(typo):
        ids.append(sp.piece_to_id(tok))
    ids.append(sp.piece_to_id("<XBC>"))
    return ids


def decode_until_xec(sp, gen_ids, xec_id):
    cut = gen_ids.index(xec_id) if xec_id in gen_ids else len(gen_ids)
    return sp.decode(gen_ids[:cut]).strip()


def categorize(typed: str, committed: str) -> str:
    """Best-effort category guess for the hold-out set."""
    if typed == committed:
        return "identity"
    nfd_t = unicodedata.normalize("NFD", typed)
    nfd_c = unicodedata.normalize("NFD", committed)
    base_t = "".join(c for c in nfd_t if not unicodedata.combining(c))
    base_c = "".join(c for c in nfd_c if not unicodedata.combining(c))
    if base_t.lower() == base_c.lower() and typed != committed:
        if "ç" in committed.lower() and "ç" not in typed.lower():
            return "cedilla_only"
        return "accent_only"
    if committed.lower().startswith(typed.lower()):
        return "prefix_completion"
    if typed.lower() == committed.lower() and typed != committed:
        return "capitalization"
    if abs(len(typed) - len(committed)) <= 1:
        return "adjacency_or_short_edit"
    return "hybrid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--beams", type=int, default=8)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    xec_id = sp.piece_to_id("<XEC>")

    eval_pairs = json.loads(Path(args.eval_jsonl).read_text())
    print(f"Loaded {len(eval_pairs)} eval pairs from {args.eval_jsonl}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    print(f"  model loaded, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    rows = []
    top1, top5 = 0, 0
    by_cat = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})

    with torch.no_grad():
        for i, pair in enumerate(eval_pairs):
            typo = pair["typed"]
            correct = pair["committed"]
            cat = categorize(typo, correct)

            prompt_ids = encode_xbu(sp, typo)
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Greedy
            out = model.generate(prompt, max_new_tokens=args.max_new, do_sample=False,
                                 num_beams=1, pad_token_id=sp.pad_id(),
                                 eos_token_id=xec_id)
            gen_ids = out[0, len(prompt_ids):].tolist()
            top1_str = decode_until_xec(sp, gen_ids, xec_id)

            # Beam
            beam_out = model.generate(prompt, max_new_tokens=args.max_new, do_sample=False,
                                      num_beams=args.beams, num_return_sequences=args.beams,
                                      pad_token_id=sp.pad_id(), eos_token_id=xec_id)
            beams = []
            for k in range(args.beams):
                ids = beam_out[k, len(prompt_ids):].tolist()
                beams.append(decode_until_xec(sp, ids, xec_id))

            t1_hit = top1_str.strip() == correct.strip()
            t5_hit = any(b.strip() == correct.strip() for b in beams[:5])
            if t1_hit: top1 += 1
            if t5_hit: top5 += 1

            by_cat[cat]["n"] += 1
            if t1_hit: by_cat[cat]["top1"] += 1
            if t5_hit: by_cat[cat]["top5"] += 1

            rows.append({
                "typed": typo, "committed": correct, "category": cat,
                "top1": top1_str, "top5": beams[:5],
                "top1_hit": t1_hit, "top5_hit": t5_hit,
            })

            if i < 15 or t1_hit or t5_hit:
                mark = "✓" if t1_hit else ("△" if t5_hit else "✗")
                print(f"  {mark} [{cat:24s}] '{typo}' → '{top1_str}' (want '{correct}')")

    n = len(eval_pairs)
    print(f"\n=== Real-typo eval: {n} pairs from {args.eval_jsonl} ===")
    print(f"  top-1: {top1}/{n} = {100.0*top1/n:.1f}%")
    print(f"  top-5: {top5}/{n} = {100.0*top5/n:.1f}%")
    print(f"\n  By category:")
    for cat, stats in sorted(by_cat.items()):
        t1 = stats["top1"] / stats["n"] * 100
        t5 = stats["top5"] / stats["n"] * 100
        print(f"    {cat:28s} n={stats['n']:3d}  top1={t1:5.1f}%  top5={t5:5.1f}%")

    Path(args.out).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "eval_jsonl": args.eval_jsonl,
        "n": n, "top1": top1, "top5": top5,
        "by_category": dict(by_cat),
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
