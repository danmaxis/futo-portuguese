#!/usr/bin/env bash
# v8.1 Claude-synth generation, via the local `claude` CLI (no API key).
#
# Requires `claude` on PATH (the Claude Code CLI; already installed locally).
# Run from the repo root.
#
# Two passes:
#   1. General (500 pairs, default priors)
#   2. Weak-spot biased (250 pairs, prefix_completion=0.40, hybrid_multi=0.25)
#
# Wall time (concurrency 8, Haiku 4.5): ~75 min for 750 accepted pairs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="notes/v8_1"
mkdir -p "$OUT_DIR"

SEED_CORPUS="${SEED_CORPUS:-corpora/big/shard_00000.txt}"
MODEL="${MODEL:-claude-haiku-4-5}"
CONCURRENCY="${CONCURRENCY:-8}"

if [ ! -f "$SEED_CORPUS" ]; then
    echo "Seed corpus not found: $SEED_CORPUS"
    echo "Set SEED_CORPUS=<path> or rsync a shard from the 3090 to corpora/big/."
    exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "\`claude\` CLI not on PATH — install Claude Code or fix PATH."
    exit 1
fi

echo "=== Pass 1: general pool (500 pairs, model=$MODEL, concurrency=$CONCURRENCY) ==="
python3 scripts/08_claude_synth.py \
    --seed-corpus "$SEED_CORPUS" \
    --out "$OUT_DIR/synth_claude_general.json" \
    --target-pairs 500 \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --seed 8101

echo
echo "=== Pass 2: weak-spot biased (250 pairs, prefix+hybrid heavy) ==="
python3 scripts/08_claude_synth.py \
    --seed-corpus "$SEED_CORPUS" \
    --out "$OUT_DIR/synth_claude_weakspot.json" \
    --target-pairs 250 \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --seed 8102 \
    --priors-json "$OUT_DIR/priors_weakspot.json"

echo
echo "Done. Outputs:"
ls -la "$OUT_DIR"/synth_claude_*.json
