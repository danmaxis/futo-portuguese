"""
Build a {word: count} JSON file from the corpus shards.

Input:  /workspace/corpora/<corpus>/shard_*.txt
Output: notes/wordfreq.json (used by 04a_finetune_isolated.py)

Why: Phase 4a's typo dataset samples words with log(freq+1) weighting per wiki.
We need an actual frequency map of the corpus we trained on.
"""
from __future__ import annotations
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

# pt_BR alpha + apostrophes/hyphens word pattern
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Dir of shard_*.txt")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--min-count", type=int, default=3, help="Drop words seen < this many times")
    ap.add_argument("--max-words", type=int, default=200_000, help="Keep top-N most frequent")
    ap.add_argument("--lowercase", action="store_true", default=True)
    args = ap.parse_args()

    shards = sorted(glob.glob(str(Path(args.corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {args.corpus}")

    print(f"Scanning {len(shards)} shards")
    counter: Counter[str] = Counter()
    for shard in shards:
        local = Counter()
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                for m in WORD_RE.finditer(line):
                    w = m.group(0)
                    if args.lowercase:
                        w = w.lower()
                    local[w] += 1
        counter.update(local)
        print(f"  {shard}: {len(local):,} unique, running total {len(counter):,}")

    print(f"Total unique words: {len(counter):,}")
    filtered = {w: n for w, n in counter.items() if n >= args.min_count}
    print(f"After min_count={args.min_count}: {len(filtered):,}")
    if len(filtered) > args.max_words:
        top = dict(sorted(filtered.items(), key=lambda kv: -kv[1])[:args.max_words])
        print(f"Trimmed to top-{args.max_words}: {len(top):,}")
    else:
        top = filtered

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(top, ensure_ascii=False))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"Top 20: {sorted(top.items(), key=lambda kv: -kv[1])[:20]}")


if __name__ == "__main__":
    main()
