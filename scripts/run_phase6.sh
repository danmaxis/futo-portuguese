#!/bin/bash
# Phase 6: Final big run on 3090 (Unraid container).
#
# Uses Phase 1's validated config:
# - stage_a base: finetune_v3/A4_plw005_full/checkpoint-1500 (16% top-5 peak)
# - stage_b: 4b with PLW=0.05, typo_rate=0.33, NO SAM (B1 winning recipe — B2/A5 showed SAM hurts)
# - stage_c: 4c on conversational corpus with same recipe
#
# Run from /workspace via:
#   nohup ./scripts/run_phase6.sh > /workspace/phase6.log 2>&1 &
#
# Idempotent: skips phases that already have final/config.json.

set -u
cd /workspace

TOK=tokenizer/spm_pt_br_v2.model
STAGE_A_BASE=finetune_v3/A4_plw005_full/checkpoint-1500
CORPUS_BIG=corpora/big
CORPUS_CONV=corpora/conv
EVAL=notes/real_typos_eval.json
LOG=/workspace/phase6.log
STATUS=/workspace/phase6.status

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

log "=== Phase 6 start ==="

# Prereq check
for f in "$TOK" "$STAGE_A_BASE/config.json" "$CORPUS_BIG" "$CORPUS_CONV" "$EVAL"; do
    if [ ! -e "$f" ]; then
        log "MISSING prereq: $f — abort"
        exit 1
    fi
done
log "prereqs OK"

# ----------------- Stage_b: in-context corrections on big corpus -----------------
if [ ! -f finetune_big_v3/stage_b/final/config.json ]; then
    log "=== Stage_b: 4b PLW=0.05 typo_rate=0.33, 15K steps ==="
    python scripts/04b_finetune_fulltext.py \
        --base "$STAGE_A_BASE" --tokenizer "$TOK" \
        --corpus "$CORPUS_BIG" \
        --plw 0.05 --typo-rate 0.33 \
        --out finetune_big_v3/stage_b \
        --total-steps 15000 --seq-len 512 \
        --micro-batch 24 --grad-accum 8 \
        --warmup 500 --lr 5e-5 --save-every 1000 --save-total-limit 20 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> /workspace/phase6_stage_b.log 2>&1
    if [ ! -f finetune_big_v3/stage_b/final/config.json ]; then
        log "Stage_b FAILED"
        exit 2
    fi
    log "Stage_b DONE"
else
    log "Stage_b already complete"
fi

# ----------------- Stage_c: conversational corpus, same recipe -----------------
if [ ! -f finetune_big_v3/stage_c/final/config.json ]; then
    log "=== Stage_c: 4c on conv corpus, PLW=0.05 typo_rate=0.10, 5K steps ==="
    python scripts/04b_finetune_fulltext.py \
        --base finetune_big_v3/stage_b/final --tokenizer "$TOK" \
        --corpus "$CORPUS_CONV" \
        --plw 0.05 --typo-rate 0.10 \
        --out finetune_big_v3/stage_c \
        --total-steps 5000 --seq-len 256 \
        --micro-batch 32 --grad-accum 4 \
        --warmup 200 --lr 2e-5 --save-every 500 --save-total-limit 15 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> /workspace/phase6_stage_c.log 2>&1
    if [ ! -f finetune_big_v3/stage_c/final/config.json ]; then
        log "Stage_c FAILED"
        exit 3
    fi
    log "Stage_c DONE"
else
    log "Stage_c already complete"
fi

# ----------------- Final eval against the 50-pair real-typo hold-out -----------------
log "=== Final eval ==="
python scripts/eval_real_typos.py \
    --checkpoint finetune_big_v3/stage_c/final \
    --tokenizer "$TOK" \
    --eval-jsonl "$EVAL" \
    --out notes/eval_phase6_stage_c.json \
    > /workspace/phase6_eval.log 2>&1 || log "eval failed (non-fatal)"

log "=== Phase 6 COMPLETE ==="
touch /workspace/phase6.done
echo "DONE" > "$STATUS"

# Summary
{
    echo
    echo "================================================================"
    echo "PHASE 6 COMPLETE at $(ts)"
    echo "================================================================"
    if [ -f notes/eval_phase6_stage_c.json ]; then
        python3 -c "
import json
f = json.load(open('notes/eval_phase6_stage_c.json'))
n = f['n']; t1 = f['top1']; t5 = f['top5']
print(f'Real-typo eval: top1={t1}/{n} ({100*t1/n:.1f}%) top5={t5}/{n} ({100*t5/n:.1f}%)')
by_cat = f.get('by_category', {})
for cat, s in sorted(by_cat.items()):
    print(f'  {cat:28s} top1={100*s[\"top1\"]/s[\"n\"]:.1f}% top5={100*s[\"top5\"]/s[\"n\"]:.1f}% (n={s[\"n\"]})')
" 2>/dev/null
    fi
    echo
    echo "Next: package as GGUF v8, push to phone (Phase 7 — needs ADB pairing)"
} >> "$LOG"
