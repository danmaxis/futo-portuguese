"""Merge v8.1 synth pools (process_typo_log 50K + Claude general + Claude weak-spot)
into a single JSON list for 04a --synth-jsonl.

Usage:
  python3 scripts/merge_synth_pools.py \\
      --inputs notes/v8_1/synth_typos.json \\
               notes/v8_1/synth_claude_general.json \\
               notes/v8_1/synth_claude_weakspot.json \\
      --out notes/v8_1/synth_combined.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged = []
    src_counts = Counter()
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        data = json.loads(p.read_text())
        for d in data:
            t = (d.get("typed") or "").strip()
            c = (d.get("committed") or "").strip()
            if not t or not c or t == c:
                continue
            d.setdefault("source", p.stem)
            merged.append(d)
            src_counts[d.get("source", p.stem)] += 1
        print(f"  loaded {len(data):>7d} from {p.name}")

    # Dedup on (typed, committed)
    seen = set()
    deduped = []
    for d in merged:
        k = (d["typed"].lower(), d["committed"].lower())
        if k in seen:
            continue
        seen.add(k)
        deduped.append(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deduped, ensure_ascii=False))
    print(f"\nWrote {len(deduped)} unique pairs (from {len(merged)} raw) to {out}")
    print("By source:")
    for src, n in src_counts.most_common():
        print(f"  {src:30s} {n}")


if __name__ == "__main__":
    main()
