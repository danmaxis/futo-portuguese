#!/usr/bin/env bash
# v8.1 training run — top-1-focused.
#
# Default strategy: redo stage_a + stage_b + stage_c with refreshed data
# (real-typo pool 335 + combined synth pool including Claude pairs). Stage_a
# is fast and benefits most from the Claude pairs.
#
# This is the 3090-container variant. Run from /workspace.
#
# Usage on 3090 container:
#   nohup ./scripts/run_v8_1.sh > /workspace/v8_1.log 2>&1 &
#
# Required env (override as needed):
#   STAGE_A=1           1 = redo stage_a with new data (default; ~20-40 min)
#   STAGE_B=1           1 = redo stage_b (default; ~2-3h)
#   STAGE_C=1           1 = redo stage_c (default; ~1-1.5h with 10K steps)
#   PLW_C=0.02          stage_c PLW (v8 was 0.05; v8.1 default is sharper)
#   PLW_B=0.05          stage_b PLW (v8-validated)
#   PLW_A=0.05          stage_a PLW (v8-validated for A4)
#   STAGE_A_STEPS=3000  v8 used checkpoint-1500; we save every 500 and let eval pick
#   STAGE_B_STEPS=12000
#   STAGE_C_STEPS=10000
#   REAL_MIX_A=0.40     stage_a real-typo mix ratio (v8 used 0.25)
#
# To skip stages, set STAGE_X=0. E.g. STAGE_A=0 STAGE_B=0 to only redo stage_c
# (control-twin recipe).

set -u
cd /workspace

TOK=tokenizer/spm_pt_br_v2.model
PRETRAIN_BASE=${PRETRAIN_BASE:-pretrain_big/checkpoint-100000}  # last pretrained ckpt; v8 stage_a came from this
STAGE_A_PREV=finetune_v3/A4_plw005_full/checkpoint-1500  # v8 stage_a head
STAGE_B_PREV=finetune_big_v3/stage_b/final               # v8 stage_b final
CORPUS_BIG=corpora/big
CORPUS_CONV=corpora/conv

# v8.1 data: synth_combined merges (synth_typos 50K + claude_general + claude_weakspot)
EVAL=notes/v8_1/real_typos_eval.json
REAL_POOL=notes/v8_1/real_typos_pool.json
SYNTH_COMBINED=notes/v8_1/synth_combined.json

VERSION=${VERSION:-v8_1}
OUT_ROOT=finetune_big_${VERSION}
LOG=/workspace/${VERSION}.log
STATUS=/workspace/${VERSION}.status

STAGE_A=${STAGE_A:-1}
STAGE_B=${STAGE_B:-1}
STAGE_C=${STAGE_C:-1}
PLW_A=${PLW_A:-0.05}
PLW_B=${PLW_B:-0.05}
PLW_C=${PLW_C:-0.02}
STAGE_A_STEPS=${STAGE_A_STEPS:-3000}
STAGE_B_STEPS=${STAGE_B_STEPS:-12000}
STAGE_C_STEPS=${STAGE_C_STEPS:-10000}
TYPO_RATE_B=${TYPO_RATE_B:-0.40}
TYPO_RATE_C=${TYPO_RATE_C:-0.15}
REAL_MIX_A=${REAL_MIX_A:-0.40}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $1" | tee -a "$LOG"; echo "$1" > "$STATUS"; }

log "=== ${VERSION} start (A=$STAGE_A B=$STAGE_B C=$STAGE_C  PLW_A=$PLW_A PLW_B=$PLW_B PLW_C=$PLW_C) ==="

for f in "$TOK" "$CORPUS_BIG" "$CORPUS_CONV" "$EVAL"; do
    if [ ! -e "$f" ]; then
        log "MISSING prereq: $f — abort"
        exit 1
    fi
done
log "prereqs OK"

# --------- Optional Stage_a redo ---------
if [ "$STAGE_A" = "1" ]; then
    if [ ! -f "$OUT_ROOT/stage_a/final/config.json" ]; then
        for f in "$PRETRAIN_BASE/config.json" "$SYNTH_COMBINED" "$REAL_POOL"; do
            [ -e "$f" ] || { log "MISSING for stage_a: $f — abort"; exit 1; }
        done
        log "=== Stage_a: PLW=$PLW_A real_mix=$REAL_MIX_A, $STAGE_A_STEPS steps ==="
        python scripts/04a_finetune_isolated.py \
            --base "$PRETRAIN_BASE" --tokenizer "$TOK" \
            --synth-jsonl "$SYNTH_COMBINED" --real-jsonl "$REAL_POOL" \
            --real-mix-ratio "$REAL_MIX_A" --plw "$PLW_A" \
            --out "$OUT_ROOT/stage_a" \
            --total-steps "$STAGE_A_STEPS" \
            --save-every 500 --save-total-limit 20 \
            --wandb-project "" \
            --eval-jsonl "$EVAL" --eval-every 500 \
            >> /workspace/${VERSION}_stage_a.log 2>&1
        if [ ! -f "$OUT_ROOT/stage_a/final/config.json" ]; then
            log "Stage_a FAILED"; exit 2
        fi
        log "Stage_a DONE"
    else
        log "Stage_a already complete"
    fi
    # Pick best stage_a checkpoint by top-5 (v8.1 stage_a uses XBU pairs only;
    # top-5 is the right gate at this stage, top-1 sharpening happens in c).
    BEST_A_STEP=$(python3 -c "
import csv
from pathlib import Path
p = Path('$OUT_ROOT/stage_a/real_typo_eval.csv')
if not p.exists(): print(''); raise SystemExit
rows = list(csv.DictReader(open(p)))
if not rows: print(''); raise SystemExit
best = max(rows, key=lambda r: (int(r['top5']), int(r['top1'])))
print(best['step'])
")
    if [ -n "$BEST_A_STEP" ] && [ -d "$OUT_ROOT/stage_a/checkpoint-$BEST_A_STEP" ]; then
        STAGE_B_BASE="$OUT_ROOT/stage_a/checkpoint-$BEST_A_STEP"
        log "Stage_a best step = $BEST_A_STEP"
    else
        STAGE_B_BASE="$OUT_ROOT/stage_a/final"
    fi
else
    STAGE_B_BASE="$STAGE_A_PREV"
    log "Stage_a skipped, stage_b base = $STAGE_B_BASE"
fi

# --------- Optional Stage_b redo ---------
if [ "$STAGE_B" = "1" ]; then
    if [ ! -f "$OUT_ROOT/stage_b/final/config.json" ]; then
        log "=== Stage_b: PLW=$PLW_B typo_rate=$TYPO_RATE_B, $STAGE_B_STEPS steps ==="
        python scripts/04b_finetune_fulltext.py \
            --base "$STAGE_B_BASE" --tokenizer "$TOK" \
            --corpus "$CORPUS_BIG" \
            --plw "$PLW_B" --typo-rate "$TYPO_RATE_B" \
            --out "$OUT_ROOT/stage_b" \
            --total-steps "$STAGE_B_STEPS" --seq-len 512 \
            --micro-batch 24 --grad-accum 8 \
            --warmup 500 --lr 5e-5 --save-every 1000 --save-total-limit 20 \
            --num-workers 2 --wandb-project "" \
            --eval-jsonl "$EVAL" --eval-every 500 \
            >> /workspace/${VERSION}_stage_b.log 2>&1
        if [ ! -f "$OUT_ROOT/stage_b/final/config.json" ]; then
            log "Stage_b FAILED"; exit 3
        fi
        log "Stage_b DONE"
        STAGE_C_BASE="$OUT_ROOT/stage_b/final"
    else
        log "Stage_b already complete"
        STAGE_C_BASE="$OUT_ROOT/stage_b/final"
    fi
else
    STAGE_C_BASE="$STAGE_B_PREV"
    log "Stage_b skipped, stage_c base = $STAGE_C_BASE"
fi

# --------- Stage_c: top-1 focused ---------
if [ "$STAGE_C" = "1" ]; then
    if [ ! -f "$OUT_ROOT/stage_c/final/config.json" ]; then
        log "=== Stage_c: PLW=$PLW_C typo_rate=$TYPO_RATE_C, $STAGE_C_STEPS steps ==="
        python scripts/04b_finetune_fulltext.py \
            --base "$STAGE_C_BASE" --tokenizer "$TOK" \
            --corpus "$CORPUS_CONV" \
            --plw "$PLW_C" --typo-rate "$TYPO_RATE_C" \
            --out "$OUT_ROOT/stage_c" \
            --total-steps "$STAGE_C_STEPS" --seq-len 256 \
            --micro-batch 32 --grad-accum 4 \
            --warmup 300 --lr 2e-5 --save-every 500 --save-total-limit 25 \
            --num-workers 2 --wandb-project "" \
            --eval-jsonl "$EVAL" --eval-every 500 \
            >> /workspace/${VERSION}_stage_c.log 2>&1
        if [ ! -f "$OUT_ROOT/stage_c/final/config.json" ]; then
            log "Stage_c FAILED"; exit 4
        fi
        log "Stage_c DONE"
    else
        log "Stage_c already complete"
    fi
fi

# --------- Pick best stage_c checkpoint by top-1 ---------
log "=== Picking best stage_c checkpoint by top-1 ==="
python3 - <<PY >> "$LOG" 2>&1
import csv, os
from pathlib import Path
csv_path = Path('${OUT_ROOT}/stage_c/real_typo_eval.csv')
if not csv_path.exists():
    print('No CSV — using final/')
    raise SystemExit
rows = list(csv.DictReader(open(csv_path)))
if not rows:
    raise SystemExit
best = max(rows, key=lambda r: (int(r['top1']), int(r['top5'])))
print(f"Best stage_c: step={best['step']}  top1={best['top1']}/{best['n']}  top5={best['top5']}/{best['n']}")
with open('${OUT_ROOT}/stage_c/BEST_STEP', 'w') as f:
    f.write(best['step'])
PY

# --------- Final eval ---------
BEST_STEP=$(cat "$OUT_ROOT/stage_c/BEST_STEP" 2>/dev/null || echo "")
if [ -n "$BEST_STEP" ] && [ -d "$OUT_ROOT/stage_c/checkpoint-$BEST_STEP" ]; then
    BEST_CKPT="$OUT_ROOT/stage_c/checkpoint-$BEST_STEP"
else
    BEST_CKPT="$OUT_ROOT/stage_c/final"
fi
log "=== Final eval on $BEST_CKPT ==="
python scripts/eval_real_typos.py \
    --checkpoint "$BEST_CKPT" --tokenizer "$TOK" \
    --eval-jsonl "$EVAL" \
    --out notes/eval_${VERSION}_stage_c.json \
    > /workspace/${VERSION}_eval.log 2>&1 || log "eval failed (non-fatal)"

# Sanity: also evaluate against the original v8 eval set (no regression check)
python scripts/eval_real_typos.py \
    --checkpoint "$BEST_CKPT" --tokenizer "$TOK" \
    --eval-jsonl notes/real_typos_eval.json \
    --out notes/eval_${VERSION}_on_v8_holdout.json \
    >> /workspace/${VERSION}_eval.log 2>&1 || log "v8-holdout eval failed (non-fatal)"

log "=== ${VERSION} COMPLETE ==="
touch /workspace/${VERSION}.done
echo "DONE" > "$STATUS"

{
    echo
    echo "================================================================"
    echo "${VERSION} COMPLETE at $(ts)  — best ckpt: $BEST_CKPT"
    echo "================================================================"
    for tag in stage_c on_v8_holdout; do
        json=notes/eval_${VERSION}_${tag}.json
        [ -f "$json" ] || continue
        echo "--- $tag ---"
        python3 -c "
import json
f = json.load(open('$json'))
n=f['n']; t1=f['top1']; t5=f['top5']
print(f'top1={t1}/{n} ({100*t1/n:.1f}%) top5={t5}/{n} ({100*t5/n:.1f}%)')
by_cat = f.get('by_category', {})
for cat, s in sorted(by_cat.items()):
    print(f'  {cat:28s} top1={100*s[\"top1\"]/s[\"n\"]:.1f}% top5={100*s[\"top5\"]/s[\"n\"]:.1f}% (n={s[\"n\"]})')
" 2>/dev/null
    done
    echo
    echo "Next: package the best checkpoint as v8.1 GGUF (scripts/run_phase7_package.sh + scripts/05_to_futo_gguf.py)"
} >> "$LOG"
