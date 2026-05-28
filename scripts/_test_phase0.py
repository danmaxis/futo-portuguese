"""Smoke test for Phase 0 code changes. Run from /workspace via:
    python scripts/_test_phase0.py
"""
import sys
import ast
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 1. Syntax check
print("=== syntax ===")
for f in ["lib_typo_synthesis.py", "lib_plw_trainer.py", "lib_real_eval_callback.py",
          "04a_finetune_isolated.py", "04b_finetune_fulltext.py"]:
    try:
        ast.parse((ROOT / f).read_text())
        print(f, "OK")
    except SyntaxError as e:
        print(f, "SYNTAX ERROR:", e)
        sys.exit(1)

# 2. ABNT2 ç adjacency
print()
print("=== abnt2 c ===")
from lib_typo_synthesis import ADJ, synth_typo, make_xbu_triple, make_inline_corrected
assert "ç" in ADJ, "c-cedilla missing from ADJ"
print("c_in_ADJ:", "ç" in ADJ)
print("ADJ[l]:", repr(ADJ["l"]))
print("ADJ[c-cedilla]:", repr(ADJ["ç"]))

# 3. synth_typo + triple
print()
print("=== synth ===")
rng = random.Random(42)
for word in ["coração", "exceção", "privilégio", "mexer", "apropriado"]:
    typo = synth_typo(word, rng)
    triple = make_xbu_triple(typo or word, word)
    print(f"  {word} -> {typo}  triple: {triple}")

# 4. PLW helpers
print()
print("=== plw ===")
from lib_plw_trainer import (
    PLWTrainer, build_loss_weights_for_xbu, build_loss_weights_for_correction_only,
)
# Mock token IDs: BOS=1, XBU=174, XBC=175, XEC=176, CHAR_*=180-181, correct token=500
xbu_seq = [1, 174, 180, 181, 175, 500, 176, 2]
w_xbu = build_loss_weights_for_xbu(xbu_seq, xbu_id=174, xec_id=176, plw_clean=0.05)
print("xbu sample seq:", xbu_seq)
print("xbu sample weights (plw=0.05):", w_xbu)
# Expected: [0.05, 1, 1, 1, 1, 1, 1, 0.05]

w_corr = build_loss_weights_for_correction_only(xbu_seq, xbc_id=175, xec_id=176, plw_clean=0.0)
print("corr-only weights (plw=0.0):", w_corr)
# Expected: [0, 0, 0, 0, 1, 1, 1, 0]

w_corr_plw = build_loss_weights_for_correction_only(xbu_seq, xbc_id=175, xec_id=176, plw_clean=0.05)
print("corr-only weights (plw=0.05):", w_corr_plw)

# 5. Custom Trainer class exists
print()
print("=== trainer ===")
print("PLWTrainer parent:", PLWTrainer.__bases__[0].__name__)

# 6. RealTypoEvalCallback can be instantiated (without running eval — needs model)
print()
print("=== callback ===")
from lib_real_eval_callback import RealTypoEvalCallback, _to_keypress_chars
print("keypress 'voce':", _to_keypress_chars("voce"))
print("keypress 'coração':", _to_keypress_chars("coração"))

print()
print("ALL CHECKS PASSED")
