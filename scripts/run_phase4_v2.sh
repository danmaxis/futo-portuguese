#!/bin/bash
# Phase 4 retrain v2 — using real-user typo dataset + calibrated synth.
#
# Continues from /workspace/pretrain_big/base (unchanged from big run).
# Outputs go to /workspace/finetune_big_v2/{stage_a,stage_b,stage_c}/.
# Idempotent — skips phases that already have final/config.json.
#
# Prereqs (pre-staged on /workspace/):
#   - pretrain_big/base/             (already there from big run)
#   - tokenizer/spm_pt_br_v2.model   (already there)
#   - corpora/big/                   (for 4b)
#   - corpora/conv/                  (for 4c)
#   - notes/synth_typos.json         (200K pairs, new)
#   - notes/real_typos_pool.json     (343 pairs, new)
#   - notes/real_typos_eval.json     (50 pairs hold-out)
#   - scripts/04a_finetune_isolated.py  (updated to read JSONL)
#   - scripts/04b_finetune_fulltext.py  (unchanged)
#
# Usage (from inside the futo-train container):
#   nohup /workspace/run_phase4_v2.sh > /workspace/run_phase4_v2.log 2>&1 &

set -u
cd /workspace

LOG=/workspace/run_phase4_v2.log
STATUS=/workspace/run_phase4_v2.status
TOK=tokenizer/spm_pt_br_v2.model
BASE=pretrain_big/base

A_DONE=finetune_big_v2/stage_a/final/config.json
B_DONE=finetune_big_v2/stage_b/final/config.json
C_DONE=finetune_big_v2/stage_c/final/config.json
EVAL_DONE=notes/eval_final_v2.json

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

log "=== Phase 4 v2 start (pid $$) ==="

for f in "$TOK" "$BASE/config.json" notes/synth_typos.json notes/real_typos_pool.json notes/real_typos_eval.json corpora/big corpora/conv; do
    if [ ! -e "$f" ]; then
        log "MISSING prereq: $f — abort"
        exit 1
    fi
done
log "prereqs OK"

# -----------------------------------------------------------------------------
# Phase 4a v2: isolated XBU triples from JSONL (synth+real mix)
# Steps reduced from 8K to 4K (real data is richer than synth alone — converges faster)
# -----------------------------------------------------------------------------
if [ ! -f "$A_DONE" ]; then
    log "=== Phase 4a v2 starting ==="
    python scripts/04a_finetune_isolated.py \
        --base "$BASE" \
        --tokenizer "$TOK" \
        --synth-jsonl notes/synth_typos.json \
        --real-jsonl notes/real_typos_pool.json \
        --real-mix-ratio 0.25 \
        --out finetune_big_v2/stage_a \
        --total-steps 4000 \
        --seq-len 64 \
        --micro-batch 64 \
        --grad-accum 4 \
        --warmup 200 \
        --lr 1e-4 \
        --save-every 1000 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4a_v2.log 2>&1
    if [ ! -f "$A_DONE" ]; then
        log "Phase 4a v2 FAILED — see /workspace/phase_4a_v2.log"
        exit 4
    fi
    log "Phase 4a v2 DONE"
else
    log "Phase 4a v2 already complete"
fi

# -----------------------------------------------------------------------------
# Phase 4b v2: in-context corrections on big corpus.
# Same script as before — typo_rate dropped to 0.10 (was 0.20 in v1) to further
# fight mode-collapse seen in the big run.
# -----------------------------------------------------------------------------
if [ ! -f "$B_DONE" ]; then
    log "=== Phase 4b v2 starting ==="
    python scripts/04b_finetune_fulltext.py \
        --base finetune_big_v2/stage_a/final \
        --tokenizer "$TOK" \
        --corpus corpora/big \
        --out finetune_big_v2/stage_b \
        --total-steps 20000 \
        --seq-len 512 \
        --micro-batch 24 \
        --grad-accum 8 \
        --warmup 500 \
        --typo-rate 0.10 \
        --save-every 2500 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4b_v2.log 2>&1
    if [ ! -f "$B_DONE" ]; then
        log "Phase 4b v2 FAILED — see /workspace/phase_4b_v2.log"
        exit 5
    fi
    log "Phase 4b v2 DONE"
else
    log "Phase 4b v2 already complete"
fi

# -----------------------------------------------------------------------------
# Phase 4c v2: conversational adaptation, very low typo_rate.
# -----------------------------------------------------------------------------
if [ ! -f "$C_DONE" ]; then
    log "=== Phase 4c v2 starting ==="
    python scripts/04b_finetune_fulltext.py \
        --base finetune_big_v2/stage_b/final \
        --tokenizer "$TOK" \
        --corpus corpora/conv \
        --out finetune_big_v2/stage_c \
        --total-steps 5000 \
        --seq-len 256 \
        --micro-batch 32 \
        --grad-accum 4 \
        --warmup 200 \
        --lr 2e-5 \
        --typo-rate 0.05 \
        --save-every 1000 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4c_v2.log 2>&1
    if [ ! -f "$C_DONE" ]; then
        log "Phase 4c v2 FAILED — see /workspace/phase_4c_v2.log"
        exit 6
    fi
    log "Phase 4c v2 DONE"
else
    log "Phase 4c v2 already complete"
fi

# -----------------------------------------------------------------------------
# Eval against REAL hold-out (notes/real_typos_eval.json)
# Need an eval-with-real script. For now, use existing eval_keyboard.py which
# uses hardcoded test cases. We'll add a real-eval mode in a follow-up.
# -----------------------------------------------------------------------------
log "=== eval (final) ==="
python scripts/eval_keyboard.py \
    --checkpoint finetune_big_v2/stage_c/final \
    --tokenizer "$TOK" \
    --out "$EVAL_DONE" \
    > /workspace/eval_v2.log 2>&1 || log "eval failed (non-fatal)"

log "=== PHASE 4 v2 COMPLETE ==="
touch /workspace/phase4_v2.done

# Summarize for the user
{
    echo "================================================================"
    echo "PHASE 4 v2 COMPLETE at $(ts)"
    echo "================================================================"
    if [ -f "$EVAL_DONE" ]; then
        python3 -c "
import json
f = json.load(open('$EVAL_DONE'))
ac, nw = f['autocorrect'], f['next_word']
print(f'autocorrect: top1={ac[\"top1\"]}/{ac[\"n\"]} top5={ac[\"top5\"]}/{ac[\"n\"]}')
print(f'next_word:   top1={nw[\"top1\"]}/{nw[\"n\"]} top8={nw[\"topk\"]}/{nw[\"n\"]}')
" 2>/dev/null
    fi
    echo
    echo "Next: pull finetune_big_v2/stage_c/final/ → build GGUF v7 → side-load"
} >> "$LOG"

echo "DONE" > "$STATUS"
exit 0
