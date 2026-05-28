#!/usr/bin/env bash
# Apply the TypoLogger patch to a fresh FUTO Keyboard checkout.
#
# Usage:
#   ./apply_patch.sh /path/to/futo-checkout
#
# Idempotent: detects already-applied patches and skips.

set -euo pipefail

REPO="${1:?usage: $0 <futo-repo-path>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO"

# 1. Drop TypoLogger.java into the utils package
DEST_DIR="java/src/org/futo/inputmethod/latin/utils"
DEST="$DEST_DIR/TypoLogger.java"
if [ -f "$DEST" ]; then
    echo "TypoLogger.java already present — overwriting"
fi
mkdir -p "$DEST_DIR"
cp "$PATCH_DIR/TypoLogger.java" "$DEST"
echo "✓ wrote $DEST"

# 2. Apply the InputLogic.java diff
if grep -q "TypoLogger.log" java/src/org/futo/inputmethod/latin/inputlogic/InputLogic.java; then
    echo "InputLogic.java already patched — skipping"
else
    patch -p1 < "$PATCH_DIR/InputLogic.diff"
    echo "✓ patched InputLogic.java"
fi

# 3. Apply the LatinIME.kt diff
if grep -q "TypoLogger.init" java/src/org/futo/inputmethod/latin/LatinIME.kt; then
    echo "LatinIME.kt already patched — skipping"
else
    patch -p1 < "$PATCH_DIR/LatinIME.diff"
    echo "✓ patched LatinIME.kt"
fi

echo
echo "All patches applied. Next: ./build_and_install.sh $REPO"
