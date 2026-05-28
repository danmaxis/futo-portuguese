#!/bin/bash
# Phase 7: Package the Phase 6 stage_c checkpoint as a FUTO-compatible GGUF v8.
#
# Steps:
#   1. Convert HF checkpoint → vanilla GGUF (via 05_to_futo_gguf.py)
#   2. Patch FUTO metadata (06_patch_metadata.py)
#   3. Downgrade GGUF metadata format to v2 (06b_downgrade_v2.py)
#   4. (manual) ADB push — requires user to pair phone
#
# Run from /workspace via:
#   bash scripts/run_phase7_package.sh
# Then user runs ADB pair + adb push manually.

set -e
cd /workspace

CKPT=${CKPT:-finetune_big_v3/stage_c/final}
TOK=${TOK:-tokenizer/spm_pt_br_v2.model}
LLAMA_CPP=${LLAMA_CPP:-/workspace/llama.cpp}
OUT_DIR=${OUT_DIR:-models}
VERSION=${VERSION:-v8}
OUT_GGUF="$OUT_DIR/futo_pt_br_${VERSION}.gguf"

mkdir -p "$OUT_DIR"

echo "=== Phase 7 packaging start ==="

if [ ! -f "$CKPT/config.json" ]; then
    echo "ERROR: Phase 6 stage_c checkpoint not found at $CKPT"
    echo "Run Phase 6 first: bash scripts/run_phase6.sh"
    exit 1
fi

echo "=== Step 1: HF → vanilla GGUF ==="
python scripts/05_to_futo_gguf.py \
    --checkpoint "$CKPT" \
    --tokenizer "$TOK" \
    --llama-cpp "$LLAMA_CPP" \
    --out "$OUT_GGUF.vanilla"

echo
echo "=== Step 2: patch FUTO metadata ==="
python scripts/06_patch_metadata.py \
    --input "$OUT_GGUF.vanilla" \
    --output "$OUT_GGUF.patched"

echo
echo "=== Step 3: downgrade GGUF to v2 (FUTO requirement) ==="
python scripts/06b_downgrade_v2.py \
    --input "$OUT_GGUF.patched" \
    --output "$OUT_GGUF"

echo
echo "=== Phase 7 packaging COMPLETE ==="
ls -lh "$OUT_GGUF"

echo
echo "Next: ADB pair + push. Run on the host that has the phone:"
echo "  adb pair <phone-ip>:<port> <pairing-code>"
echo "  adb connect <phone-ip>:<port>"
echo "  adb push $OUT_GGUF /sdcard/Android/data/org.futo.inputmethod.latin.playstore/files/models/"
