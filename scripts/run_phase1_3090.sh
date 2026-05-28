#!/bin/bash
# Phase 1 ablations on 3090 (Unraid container): A1 → A2 sequentially.
# Run from /workspace via:
#   nohup ./scripts/run_phase1_3090.sh > /workspace/phase1_3090.log 2>&1 &
#
# Both runs are 5K steps of 04a, distinguished only by --plw value.
# Eval every 500 steps writes top-1/top-5 to {out}/real_typo_eval.csv.

set -u
cd /workspace

TOK=tokenizer/spm_pt_br_v2.model
BASE=pretrain_big/base
SYNTH=notes/synth_typos.json
REAL=notes/real_typos_pool.json
EVAL=notes/real_typos_eval.json
LOG=/workspace/phase1_3090.log
STATUS=/workspace/phase1_3090.status

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

log "=== Phase 1 3090 start ==="

# A1: 4a PLW=0 (baseline)
if [ ! -f finetune_v3/A1_plw0/final/config.json ]; then
    log "=== A1: 04a PLW=0 ==="
    python scripts/04a_finetune_isolated.py \
        --base "$BASE" --tokenizer "$TOK" \
        --synth-jsonl "$SYNTH" --real-jsonl "$REAL" \
        --real-mix-ratio 0.25 --plw 0.0 \
        --out finetune_v3/A1_plw0 \
        --total-steps 5000 --seq-len 64 \
        --micro-batch 32 --grad-accum 1 \
        --warmup 200 --lr 1e-4 --save-every 5000 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> /workspace/phase1_A1.log 2>&1
    log "=== A1 done ==="
else
    log "A1 already done"
fi

# A2: 4a PLW=0.05
if [ ! -f finetune_v3/A2_plw005/final/config.json ]; then
    log "=== A2: 04a PLW=0.05 ==="
    python scripts/04a_finetune_isolated.py \
        --base "$BASE" --tokenizer "$TOK" \
        --synth-jsonl "$SYNTH" --real-jsonl "$REAL" \
        --real-mix-ratio 0.25 --plw 0.05 \
        --out finetune_v3/A2_plw005 \
        --total-steps 5000 --seq-len 64 \
        --micro-batch 32 --grad-accum 1 \
        --warmup 200 --lr 1e-4 --save-every 5000 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> /workspace/phase1_A2.log 2>&1
    log "=== A2 done ==="
else
    log "A2 already done"
fi

log "=== Phase 1 3090 COMPLETE ==="
touch /workspace/phase1_3090.done
echo "DONE" > "$STATUS"
