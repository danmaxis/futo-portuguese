"""
Conversational NWP + prefix-completion eval. Complements eval_real_typos.py
(which only measures typo-correction) by testing what the user actually sees
mid-sentence in real typing.

For each word boundary in each conversation:
  - NWP: feed context, model generates next token(s) without XBU prompt. Check
    whether the actual next word appears in top-1 / top-3 / top-5.
  - prefix-completion: feed `<XBU>PREFIX<XBC>` where PREFIX is the first 2-3
    keypress characters of the next word. Check top-1 / top-3 / top-5 for the
    full word.

Output: per-scenario and overall hit rates, plus a per-conversation breakdown
for spot-checking. Designed to be run on v8 (baseline) and v8.1 (candidate)
back-to-back on the same convo set.

Usage:
  python3 scripts/eval_conversational.py \\
      --checkpoint finetune_big_v8_1/stage_c/final \\
      --tokenizer tokenizer/spm_pt_br_v2.model \\
      --conversations notes/v8_1/conversations.json \\
      --out notes/v8_1/eval_conv_v8_1.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM


WORD_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


def to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        if base in ("ç", "Ç"):
            base = "C"
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out


def encode_xbu_prefix(sp, prefix: str, xbu_id: int, xbc_id: int) -> list[int]:
    ids = [xbu_id]
    for tok in to_keypress_chars(prefix):
        tid = sp.piece_to_id(tok)
        if tid > 0:
            ids.append(tid)
    ids.append(xbc_id)
    return ids


def decode_until_word_break(sp, gen_ids: list[int], xec_id: int) -> str:
    """Decode generated ids until XEC, whitespace, or punctuation."""
    out_pieces = []
    for tid in gen_ids:
        if tid == xec_id:
            break
        out_pieces.append(tid)
    if not out_pieces:
        return ""
    text = sp.decode(out_pieces).strip()
    # Take just the first whitespace/punct-delimited word
    m = WORD_RE.search(text)
    return m.group(0) if m else ""


def conversations_to_running_text(convs: list[dict]) -> list[tuple[str, str, list[str]]]:
    """For each conversation, return (scenario, full_text, word_list).
    word_list is the sequence of words; positions index into full_text."""
    out = []
    for conv in convs:
        scenario = conv.get("scenario", "unknown")
        parts = [m["text"].strip() for m in conv["messages"] if m.get("text", "").strip()]
        full = " ".join(parts)
        words = WORD_RE.findall(full)
        out.append((scenario, full, words))
    return out


def context_before_word(full: str, word_idx: int, words: list[str]) -> str:
    """Return the running text up to (but not including) the word at word_idx."""
    if word_idx == 0:
        return ""
    # Find the start of the word_idx-th word in full
    cursor = 0
    found = 0
    for m in WORD_RE.finditer(full):
        if found == word_idx:
            return full[:m.start()].rstrip()
        found += 1
    return full


def top_k_nwp(model, sp, context: str, k: int = 5,
              max_new: int = 8, bad_words_ids: list[list[int]] | None = None) -> list[str]:
    """Run beam search on `context` and return up to k candidate next words.
    bad_words_ids: list of token-id sequences to never emit (e.g., XBU/CHAR_*
    special tokens — these are not surface words and pollute NWP top-k)."""
    device = next(model.parameters()).device
    bos = sp.bos_id()
    ids = [bos] + sp.encode(context, out_type=int)
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            prompt, max_new_tokens=max_new, do_sample=False,
            num_beams=k, num_return_sequences=k,
            pad_token_id=sp.pad_id(),
            bad_words_ids=bad_words_ids,
        )
    cands = []
    for j in range(out.size(0)):
        tail = out[j, len(ids):].tolist()
        w = decode_until_word_break(sp, tail, xec_id=-1)  # no xec in NWP
        if w:
            cands.append(w)
    # Dedup preserving order
    seen, dedup = set(), []
    for w in cands:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl); dedup.append(w)
    return dedup[:k]


def top_k_prefix(model, sp, context: str, prefix: str,
                 xbu_id: int, xbc_id: int, xec_id: int,
                 k: int = 5, max_new: int = 12) -> list[str]:
    """Predict a word completion given XBU prefix. Returns up to k candidates."""
    device = next(model.parameters()).device
    bos = sp.bos_id()
    ctx_ids = [bos] + sp.encode(context + " " if context else "", out_type=int)
    xbu_ids = encode_xbu_prefix(sp, prefix, xbu_id, xbc_id)
    full_ids = ctx_ids + xbu_ids
    prompt = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            prompt, max_new_tokens=max_new, do_sample=False,
            num_beams=k, num_return_sequences=k,
            pad_token_id=sp.pad_id(), eos_token_id=xec_id,
        )
    cands = []
    for j in range(out.size(0)):
        tail = out[j, len(full_ids):].tolist()
        w = decode_until_word_break(sp, tail, xec_id=xec_id)
        if w:
            cands.append(w)
    seen, dedup = set(), []
    for w in cands:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl); dedup.append(w)
    return dedup[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--skip-prefix", action="store_true",
                    help="Only run NWP, skip prefix-completion (faster).")
    ap.add_argument("--max-words-per-conv", type=int, default=60,
                    help="Cap to keep runtime predictable.")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = LlamaForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    xbu_id = sp.piece_to_id("<XBU>")
    xbc_id = sp.piece_to_id("<XBC>")
    xec_id = sp.piece_to_id("<XEC>")

    # Special tokens to ban from NWP beam search — these are not surface words.
    # Without this mask, the model emits <XBU> as top-1 ~100% of the time
    # because typo training taught it that XBU is a valid continuation after
    # any word. FUTO inference never proposes special tokens to the user, so
    # masking here mirrors production behavior.
    nwp_bad_words = [[xbu_id], [xbc_id], [xec_id]]
    for c in range(ord("A"), ord("Z") + 1):
        tid = sp.piece_to_id(f"<CHAR_{chr(c)}>")
        if tid > 0:
            nwp_bad_words.append([tid])

    convs = json.loads(Path(args.conversations).read_text())
    print(f"Loaded {len(convs)} conversations", flush=True)

    # Aggregate counters
    nwp = {"top1": 0, "top3": 0, "top5": 0, "n": 0}
    pfx = {"top1": 0, "top3": 0, "top5": 0, "n": 0}
    by_scenario = defaultdict(lambda: {"nwp": dict(nwp), "pfx": dict(pfx)})
    samples = []  # for spot-check

    for ci, conv in enumerate(convs):
        scenario = conv.get("scenario", "unknown")
        parts = [m["text"].strip() for m in conv["messages"] if m.get("text", "").strip()]
        full = " ".join(parts)
        words = WORD_RE.findall(full)
        if not words:
            continue
        words = words[:args.max_words_per_conv]

        for wi in range(1, len(words)):  # skip first word (no prior context)
            actual = words[wi]
            actual_l = actual.lower()
            context = context_before_word(full, wi, words)

            # NWP
            try:
                cands = top_k_nwp(model, sp, context, k=args.beams,
                                  bad_words_ids=nwp_bad_words)
            except Exception as e:
                print(f"  conv{ci} word{wi} NWP error: {e}", flush=True)
                cands = []
            cands_l = [c.lower() for c in cands]
            nwp["n"] += 1
            by_scenario[scenario]["nwp"]["n"] += 1
            for k in (1, 3, 5):
                if actual_l in cands_l[:k]:
                    nwp[f"top{k}"] += 1
                    by_scenario[scenario]["nwp"][f"top{k}"] += 1

            if len(samples) < 30 and wi <= 8:
                samples.append({
                    "scenario": scenario, "conv": ci, "wi": wi,
                    "context_tail": context[-60:], "actual": actual,
                    "nwp_top5": cands,
                })

            # Prefix completion: take first 2 and 3 chars of actual word
            if args.skip_prefix:
                continue
            for plen in (2, 3):
                if len(actual) <= plen:
                    continue
                prefix = actual[:plen]
                try:
                    pcands = top_k_prefix(model, sp, context, prefix,
                                          xbu_id, xbc_id, xec_id, k=args.beams)
                except Exception as e:
                    print(f"  conv{ci} word{wi} PFX error: {e}", flush=True)
                    pcands = []
                pcands_l = [c.lower() for c in pcands]
                pfx["n"] += 1
                by_scenario[scenario]["pfx"]["n"] += 1
                for k in (1, 3, 5):
                    if actual_l in pcands_l[:k]:
                        pfx[f"top{k}"] += 1
                        by_scenario[scenario]["pfx"][f"top{k}"] += 1

        if (ci + 1) % 5 == 0:
            print(f"  done conv {ci+1}/{len(convs)} — nwp_top3={nwp['top3']}/{nwp['n']} "
                  f"pfx_top3={pfx['top3']}/{pfx['n']}", flush=True)

    def pct(d):
        return {k: (100 * d[k] / d["n"] if d["n"] else 0.0) for k in ("top1", "top3", "top5")} | {"n": d["n"]}

    result = {
        "checkpoint": args.checkpoint,
        "n_conversations": len(convs),
        "nwp": pct(nwp),
        "prefix": pct(pfx),
        "by_scenario": {s: {"nwp": pct(d["nwp"]), "pfx": pct(d["pfx"])} for s, d in by_scenario.items()},
        "samples": samples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== Conversational eval ===")
    print(f"NWP        top1={result['nwp']['top1']:5.1f}%  top3={result['nwp']['top3']:5.1f}%  "
          f"top5={result['nwp']['top5']:5.1f}%  (n={result['nwp']['n']})")
    if not args.skip_prefix:
        print(f"Prefix     top1={result['prefix']['top1']:5.1f}%  top3={result['prefix']['top3']:5.1f}%  "
              f"top5={result['prefix']['top5']:5.1f}%  (n={result['prefix']['n']})")
    print("\nBy scenario:")
    for s in sorted(by_scenario):
        r = result["by_scenario"][s]
        print(f"  {s:22s} nwp_top3={r['nwp']['top3']:5.1f}%  pfx_top3={r['pfx']['top3']:5.1f}%")


if __name__ == "__main__":
    main()
