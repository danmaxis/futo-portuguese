#!/bin/bash
# Phase 1 ablations on gpu-5070ti: B1 → B2 sequentially.
# Run from ~/futo-train via:
#   nohup ./scripts/run_phase1_5070ti.sh > ~/futo-train/phase1_5070ti.log 2>&1 &
#
# B1 is the critical run: 4b at PLW=0.05 (the fix), typo_rate=0.33 (wiki value).
# Stop-go criterion: B1 must hit top-5 ≥ 12% by step 5K to proceed to Phase 2.
# B2 = B1 + SAM (doubles per-step cost).

set -u
cd "$HOME/futo-train"
export PATH="$HOME/.local/bin:$PATH"
PY="$HOME/futo-train/.venv/bin/python"

TOK=tokenizer/spm_pt_br_v2.model
BASE=pretrain_big/base
CORPUS=corpora/big
EVAL=notes/real_typos_eval.json
LOG=$HOME/futo-train/phase1_5070ti.log
STATUS=$HOME/futo-train/phase1_5070ti.status

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

log "=== Phase 1 5070 Ti start ==="

# Need a stage_a checkpoint for 04b's --base. Use pretrain_big/base directly for
# the ablation (skip the 4a step locally — the goal is to test the PLW fix in
# 4b's regime, not the full pipeline).
B_BASE="$BASE"

# B1: 04b PLW=0.05, typo_rate=0.33 — THE critical run
if [ ! -f finetune_v3/B1_plw005/final/config.json ]; then
    log "=== B1: 04b PLW=0.05 typo_rate=0.33 ==="
    "$PY" scripts/04b_finetune_fulltext.py \
        --base "$B_BASE" --tokenizer "$TOK" \
        --corpus "$CORPUS" \
        --plw 0.05 --typo-rate 0.33 \
        --out finetune_v3/B1_plw005 \
        --total-steps 5000 --seq-len 512 \
        --micro-batch 12 --grad-accum 4 \
        --warmup 200 --lr 5e-5 --save-every 5000 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> $HOME/futo-train/phase1_B1.log 2>&1
    log "=== B1 done ==="
else
    log "B1 already done"
fi

# B2: B1 + SAM
if [ ! -f finetune_v3/B2_plw005_sam/final/config.json ]; then
    log "=== B2: 04b PLW=0.05 typo_rate=0.33 + SAM ==="
    "$PY" scripts/04b_finetune_fulltext.py \
        --base "$B_BASE" --tokenizer "$TOK" \
        --corpus "$CORPUS" \
        --plw 0.05 --typo-rate 0.33 \
        --use-sam --sam-rho 0.05 \
        --out finetune_v3/B2_plw005_sam \
        --total-steps 5000 --seq-len 512 \
        --micro-batch 12 --grad-accum 4 \
        --warmup 200 --lr 5e-5 --save-every 5000 \
        --num-workers 2 --wandb-project "" \
        --eval-jsonl "$EVAL" --eval-every 500 \
        >> $HOME/futo-train/phase1_B2.log 2>&1
    log "=== B2 done ==="
else
    log "B2 already done"
fi

log "=== Phase 1 5070 Ti COMPLETE ==="
touch $HOME/futo-train/phase1_5070ti.done
echo "DONE" > "$STATUS"
