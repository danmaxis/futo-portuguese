"""
Phase 1: assemble a Brazilian Portuguese corpus.

Streams from a configurable set of HuggingFace datasets, filters to pt_BR,
applies light cleaning + dedup, and writes plain-text shards (~256 MB each).

Designed to run on the GPU host (5070 Ti or 3090 container) — it's I/O-heavy
and writes 5-30 GB to /workspace/corpora. Streaming mode avoids ever
downloading the full datasets locally.

Usage:
  python 01_build_corpus.py --out /workspace/corpora/clean --target-tokens 3_000_000_000

  # Just pull conversational corpus for stage-4c fine-tune:
  python 01_build_corpus.py --datasets conversational --out /workspace/corpora/conversational --target-tokens 100_000_000
"""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
from pathlib import Path

# Datasets registry. Each entry: name -> (HF id, config, split, text_key, lang_filter)
# `lang_filter` is a callable(doc) -> bool when the dataset mixes languages.
PRIMARY_DATASETS = {
    "wikipedia_pt": dict(
        repo="wikimedia/wikipedia",
        config="20231101.pt",  # pt covers both pt-BR and pt-PT; we keep all and bias by frequency
        split="train",
        text_key="text",
        lang_filter=None,
    ),
    "brwac": dict(
        repo="eduagarcia/BrWac",
        config=None,
        split="train",
        text_key="text",
        lang_filter=None,  # already pt-BR
    ),
    "carolina": dict(
        repo="carolina-c4ai/Carolina-Open",
        config=None,
        split="train",
        text_key="text",
        lang_filter=None,
    ),
    "oscar_pt": dict(
        repo="oscar-corpus/OSCAR-2301",
        config="pt",
        split="train",
        text_key="text",
        lang_filter=None,
    ),
    "mc4_pt": dict(
        repo="allenai/c4",
        config="pt",
        split="train",
        text_key="text",
        lang_filter=None,
    ),
}

# Conversational / casual pt_BR — these are the GOOD signal for a phone keyboard.
# Wikipedia/news is formal; people don't type encyclopedia entries on their phone.
# OpenSubtitles is the strongest scalable casual source (film/TV dialogue dubbed
# or originally in pt_BR). CORAA and Common Voice add transcribed real speech.
CONVERSATIONAL_DATASETS = {
    "opensubtitles_pt_br": dict(
        # OPUS OpenSubtitles — millions of subtitle lines, pt-BR config
        repo="Helsinki-NLP/opus-100",
        config="en-pt",  # we keep the pt side; the other half is dropped by lang_filter
        split="train",
        text_key="translation",  # dict {'en': ..., 'pt': ...}; handled by stream_dataset hook
        lang_filter=None,
        min_chars=20,  # subtitle lines are short
    ),
    "open_subtitles_alt": dict(
        # Alternative: the legacy `open_subtitles` HF dataset has direct pt-BR docs.
        repo="open_subtitles",
        config="pt-BR",
        split="train",
        text_key="text",
        lang_filter=None,
        min_chars=20,
    ),
    "common_voice_pt": dict(
        # Common Voice transcripts — short, casual, real-speech sentences
        repo="mozilla-foundation/common_voice_17_0",
        config="pt",
        split="train",
        text_key="sentence",
        lang_filter=None,
        min_chars=15,
    ),
    "coraa_ptbr": dict(
        # CORAA — Brazilian Portuguese conversational speech transcripts
        repo="gabrielrstan/CORAA-v1.1",
        config=None,
        split="train",
        text_key="text",
        lang_filter=None,
        min_chars=15,
    ),
}

# "Big run" preset: primary + conversational, balanced. The conversational
# sources contribute disproportionately to vocabulary and tone for a keyboard,
# so we slightly upweight them relative to their token count by including them
# in full while capping the formal sources.
BIG_RUN_DATASETS = {**PRIMARY_DATASETS, **CONVERSATIONAL_DATASETS}

# Heuristic pt-BR vs pt-PT discriminator for noisy multi-locale datasets.
# Brazilian Portuguese tends to use "você" over "tu", drops the syllabic é/â pattern
# in some endings, and uses certain spelling differences. This is a soft bias — we
# keep both but slightly downweight strongly-PT docs.
PT_PT_MARKERS = re.compile(r"\b(estás|sapatilhas|telemóvel|autocarro|ginásio|comboio)\b", re.I)
PT_BR_MARKERS = re.compile(r"\b(você|ônibus|trem|celular|geladeira|caminhão|legal\s|massa\s|valeu)\b", re.I)
NON_LATIN_RATIO_LIMIT = 0.5
MIN_DOC_CHARS = 100
HASH_PREFIX = 200  # first N chars used for dedup hash


def is_likely_ptbr(text: str) -> bool:
    """Soft filter: keep all pt unless strongly pt-PT and not pt-BR."""
    pt_pt = len(PT_PT_MARKERS.findall(text))
    pt_br = len(PT_BR_MARKERS.findall(text))
    if pt_pt > 3 and pt_br == 0:
        return False
    return True


def is_clean(text: str, min_chars: int = MIN_DOC_CHARS) -> bool:
    if len(text) < min_chars:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) and (non_ascii / len(text)) > 0.7:
        # Mostly non-ASCII (e.g. CJK leakage in OSCAR) — drop.
        return False
    return True


def normalize(text: str) -> str:
    # Strip common HTML residue, collapse whitespace, but preserve casing/punct/accents.
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def doc_hash(text: str) -> str:
    return hashlib.blake2b(text[:HASH_PREFIX].encode("utf-8"), digest_size=8).hexdigest()


def shard_writer(out_dir: Path, shard_target_bytes: int = 256 * 1024 * 1024):
    """Yield a (write_fn, close_fn) generator that rotates shard files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    f = None
    bytes_written = 0

    def open_shard():
        nonlocal f, bytes_written, shard_idx
        path = out_dir / f"shard_{shard_idx:05d}.txt"
        f = open(path, "w", encoding="utf-8")
        bytes_written = 0
        return path

    def write(text: str):
        nonlocal f, bytes_written, shard_idx
        if f is None or bytes_written >= shard_target_bytes:
            if f is not None:
                f.close()
                shard_idx += 1
            open_shard()
        line = text + "\n"
        f.write(line)
        bytes_written += len(line.encode("utf-8"))

    def close():
        if f is not None:
            f.close()

    return write, close


def _extract_text(ex: dict, key: str) -> str:
    """Get the text from one example. Handles plain strings + opus-100-style
    {'translation': {'en': '...', 'pt': '...'}} layouts."""
    val = ex.get(key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # opus-100 / parallel-corpus pattern: pull pt or pt_br field
        for lang in ("pt", "pt_br", "pt-BR", "ptb"):
            if lang in val and isinstance(val[lang], str):
                return val[lang]
        # fallback: first string-valued entry
        for v in val.values():
            if isinstance(v, str):
                return v
    return ""


def stream_dataset(name: str, conf: dict, hf_token: str | None = None):
    """Yield cleaned, pt_BR-likely documents from a single HF dataset, streaming."""
    from datasets import load_dataset
    print(f"[{name}] loading streaming dataset {conf['repo']} (config={conf['config']!r})", file=sys.stderr)
    kwargs = dict(streaming=True, split=conf["split"])
    if conf["config"]:
        kwargs["name"] = conf["config"]
    if hf_token:
        kwargs["token"] = hf_token
    try:
        ds = load_dataset(conf["repo"], **kwargs)
    except Exception as e:
        print(f"[{name}] FAILED to load: {type(e).__name__}: {e}", file=sys.stderr)
        return
    seen_hashes: set[str] = set()
    kept = dropped = 0
    # Casual sources (subtitles, voice transcripts) tend to be 1-line short utterances
    min_chars = conf.get("min_chars", MIN_DOC_CHARS)
    for ex in ds:
        text = _extract_text(ex, conf["text_key"])
        if not text:
            continue
        text = normalize(text)
        if not is_clean(text, min_chars=min_chars):
            dropped += 1
            continue
        if not is_likely_ptbr(text):
            dropped += 1
            continue
        h = doc_hash(text)
        if h in seen_hashes:
            dropped += 1
            continue
        seen_hashes.add(h)
        kept += 1
        if kept % 10_000 == 0:
            print(f"[{name}] kept={kept} dropped={dropped} hashes={len(seen_hashes)}", file=sys.stderr)
        yield text


def estimate_tokens(text: str) -> int:
    # Cheap byte-pair approximation: ~4 chars per token for romance languages.
    return max(1, len(text) // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for shard files")
    ap.add_argument("--target-tokens", type=int, default=3_000_000_000,
                    help="Approximate token budget; stop after reaching it")
    ap.add_argument("--datasets", choices=["primary", "conversational", "big"], default="primary",
                    help="Which dataset bundle to pull. 'big' = primary + conversational for full pretrain run.")
    ap.add_argument("--shard-bytes", type=int, default=256 * 1024 * 1024,
                    help="Target bytes per shard file")
    ap.add_argument("--datasets-only", nargs="*", default=None,
                    help="Optional whitelist of dataset names from the registry")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.datasets == "primary":
        registry = PRIMARY_DATASETS
    elif args.datasets == "conversational":
        registry = CONVERSATIONAL_DATASETS
    else:  # "big"
        registry = BIG_RUN_DATASETS
    if args.datasets_only:
        registry = {k: v for k, v in registry.items() if k in args.datasets_only}
        if not registry:
            sys.exit("No matching datasets in registry")

    write, close = shard_writer(out_dir, shard_target_bytes=args.shard_bytes)
    total_tokens = 0
    try:
        for name, conf in registry.items():
            if total_tokens >= args.target_tokens:
                break
            for text in stream_dataset(name, conf):
                write(text)
                total_tokens += estimate_tokens(text)
                if total_tokens >= args.target_tokens:
                    print(f"[done] reached token budget {total_tokens:,}", file=sys.stderr)
                    break
    finally:
        close()

    print(f"[final] total_tokens≈{total_tokens:,} out={out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
