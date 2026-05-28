#!/bin/bash
# Master orchestration for the big run.
# Runs autonomously for 5-9 days. Idempotent — checks for done markers per phase
# and skips completed phases. Safe to re-launch if interrupted.
#
# Required pre-staged on /workspace/:
#   - tokenizer/spm_pt_br_v2.model
#   - corpora/big/   (62 shards)
#   - corpora/conv/  (12M tokens conversational)
#   - scripts/       (all training scripts)
#
# Usage (from inside the futo-train container):
#   nohup /workspace/run_big_run.sh > /workspace/run_big_run.log 2>&1 &

set -u  # strict: no undefined vars
cd /workspace

LOG=/workspace/run_big_run.log
STATUS=/workspace/run_big_run.status
TOKENIZER=tokenizer/spm_pt_br_v2.model

# Phase done markers (HF Trainer writes config.json on save_model)
B_DONE=pretrain_big/base/config.json
A_DONE=finetune_big/stage_a/final/config.json
BB_DONE=finetune_big/stage_b/final/config.json
C_DONE=finetune_big/stage_c/final/config.json
EVAL_DONE=notes/eval_final_big.json

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
log "=== big run start (pid $$) ==="
log "checking prereqs"
for f in "$TOKENIZER" corpora/big corpora/conv scripts/03_pretrain.py scripts/04a_finetune_isolated.py scripts/04b_finetune_fulltext.py scripts/04a_build_wordfreq.py scripts/eval_keyboard.py; do
    if [ ! -e "$f" ]; then
        log "MISSING prereq: $f — abort"
        exit 1
    fi
done
log "prereqs OK"

mkdir -p notes

# -----------------------------------------------------------------------------
# Phase B: broad pretrain (~5-7 days on 3090)
# -----------------------------------------------------------------------------
if [ ! -f "$B_DONE" ]; then
    log "=== Phase B (pretrain) starting ==="
    python scripts/03_pretrain.py \
        --tokenizer "$TOKENIZER" \
        --corpus corpora/big \
        --out pretrain_big \
        --total-steps 100000 \
        --seq-len 1024 \
        --micro-batch 16 \
        --grad-accum 16 \
        --warmup 2000 \
        --save-every 5000 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_B.log 2>&1
    if [ ! -f "$B_DONE" ]; then
        log "Phase B FAILED — check /workspace/phase_B.log"
        exit 2
    fi
    log "Phase B DONE"
else
    log "Phase B already complete (skipping)"
fi

# -----------------------------------------------------------------------------
# Wordfreq for Phase 4a
# -----------------------------------------------------------------------------
WF=notes/wordfreq_big.json
if [ ! -f "$WF" ]; then
    log "=== building wordfreq ==="
    python scripts/04a_build_wordfreq.py \
        --corpus corpora/big \
        --out "$WF" \
        --min-count 5 \
        --max-words 100000 \
        > /workspace/wordfreq.log 2>&1
    if [ ! -f "$WF" ]; then
        log "wordfreq build FAILED"
        exit 3
    fi
    log "wordfreq DONE ($(wc -c < $WF) bytes)"
fi

# -----------------------------------------------------------------------------
# Phase 4a: isolated XBU triples (~20 min)
# -----------------------------------------------------------------------------
if [ ! -f "$A_DONE" ]; then
    log "=== Phase 4a starting ==="
    python scripts/04a_finetune_isolated.py \
        --base pretrain_big/base \
        --tokenizer "$TOKENIZER" \
        --word-freq "$WF" \
        --out finetune_big/stage_a \
        --total-steps 8000 \
        --seq-len 64 \
        --micro-batch 64 \
        --grad-accum 4 \
        --warmup 300 \
        --typos-per-word 2 \
        --save-every 2000 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4a.log 2>&1
    if [ ! -f "$A_DONE" ]; then
        log "Phase 4a FAILED — check /workspace/phase_4a.log"
        exit 4
    fi
    log "Phase 4a DONE"
else
    log "Phase 4a already complete (skipping)"
fi

# -----------------------------------------------------------------------------
# Phase 4b: in-context corrections on big corpus (~5-6h)
# -----------------------------------------------------------------------------
if [ ! -f "$BB_DONE" ]; then
    log "=== Phase 4b starting ==="
    python scripts/04b_finetune_fulltext.py \
        --base finetune_big/stage_a/final \
        --tokenizer "$TOKENIZER" \
        --corpus corpora/big \
        --out finetune_big/stage_b \
        --total-steps 25000 \
        --seq-len 512 \
        --micro-batch 24 \
        --grad-accum 8 \
        --warmup 500 \
        --typo-rate 0.20 \
        --save-every 2500 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4b.log 2>&1
    if [ ! -f "$BB_DONE" ]; then
        log "Phase 4b FAILED — check /workspace/phase_4b.log"
        exit 5
    fi
    log "Phase 4b DONE"
else
    log "Phase 4b already complete (skipping)"
fi

# -----------------------------------------------------------------------------
# Phase 4c: casual register adaptation on conv corpus (~1h)
# -----------------------------------------------------------------------------
if [ ! -f "$C_DONE" ]; then
    log "=== Phase 4c starting ==="
    python scripts/04b_finetune_fulltext.py \
        --base finetune_big/stage_b/final \
        --tokenizer "$TOKENIZER" \
        --corpus corpora/conv \
        --out finetune_big/stage_c \
        --total-steps 8000 \
        --seq-len 256 \
        --micro-batch 32 \
        --grad-accum 4 \
        --warmup 200 \
        --lr 2e-5 \
        --typo-rate 0.10 \
        --save-every 2000 \
        --num-workers 2 \
        --wandb-project "" \
        > /workspace/phase_4c.log 2>&1
    if [ ! -f "$C_DONE" ]; then
        log "Phase 4c FAILED — check /workspace/phase_4c.log"
        exit 6
    fi
    log "Phase 4c DONE"
else
    log "Phase 4c already complete (skipping)"
fi

# -----------------------------------------------------------------------------
# Eval: floor (post-pretrain) + final (post-Phase-4c)
# -----------------------------------------------------------------------------
log "=== eval (floor) ==="
python scripts/eval_keyboard.py \
    --checkpoint pretrain_big/base \
    --tokenizer "$TOKENIZER" \
    --out notes/eval_floor_big.json \
    > /workspace/eval_floor.log 2>&1 || log "eval floor failed (non-fatal)"

log "=== eval (final) ==="
python scripts/eval_keyboard.py \
    --checkpoint finetune_big/stage_c/final \
    --tokenizer "$TOKENIZER" \
    --out "$EVAL_DONE" \
    > /workspace/eval_final.log 2>&1 || log "eval final failed (non-fatal)"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log "=== BIG RUN COMPLETE ==="
touch /workspace/big_run.done

# Summarize for the user when they come back
{
    echo "================================================================"
    echo "BIG RUN COMPLETE at $(ts)"
    echo "================================================================"
    echo
    if [ -f "notes/eval_floor_big.json" ]; then
        python3 -c "
import json
f = json.load(open('notes/eval_floor_big.json'))
ac, nw = f['autocorrect'], f['next_word']
print(f'FLOOR: autocorrect top1={ac[\"top1\"]}/{ac[\"n\"]} top5={ac[\"top5\"]}/{ac[\"n\"]}')
print(f'       next_word top1={nw[\"top1\"]}/{nw[\"n\"]} top8={nw[\"topk\"]}/{nw[\"n\"]}')
" 2>/dev/null || true
    fi
    if [ -f "$EVAL_DONE" ]; then
        python3 -c "
import json
f = json.load(open('$EVAL_DONE'))
ac, nw = f['autocorrect'], f['next_word']
print(f'FINAL: autocorrect top1={ac[\"top1\"]}/{ac[\"n\"]} top5={ac[\"top5\"]}/{ac[\"n\"]}')
print(f'       next_word top1={nw[\"top1\"]}/{nw[\"n\"]} top8={nw[\"topk\"]}/{nw[\"n\"]}')
" 2>/dev/null || true
    fi
    echo
    echo "Artifacts to package into GGUF:"
    echo "  - finetune_big/stage_c/final/   (HF checkpoint)"
    echo "  - tokenizer/spm_pt_br_v2.model"
    echo
    echo "Next: pull these to the build machine, run scripts/05_to_futo_gguf.py"
    echo "      then llama-quantize --output-tensor-type q6_k"
    echo "      then scripts/06b_downgrade_v2.py"
    echo "      then push to phone."
} >> "$LOG"

echo "DONE" > "$STATUS"
exit 0
