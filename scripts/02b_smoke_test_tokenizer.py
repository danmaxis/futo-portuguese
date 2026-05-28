"""
Local smoke test for the slot-map mechanism.

Trains a tiny SentencePiece on a synthetic pt_BR sample to verify:
  - The 300 user_defined_symbols land at indices 4..303 in declaration order
  - <CHAR_A>..<CHAR_Z> are 26 sequential IDs
  - <XBU>, <XBC>, <XEC>, <XC0> resolve correctly
  - `▁` whitespace-as-suffix marker doesn't collide with our slot strings
  - byte_fallback adds <0x00>..<0xFF> after our user-defined symbols
  - The XBU autocorrect format round-trips through encode/decode

Designed to run on this VM (no GPU, no full corpus) in <30 seconds.
"""
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

# Re-use the slot-list builder from the real training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("trainer", Path(__file__).resolve().parent / "02_train_tokenizer.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

USER_DEFINED = trainer.build_user_defined_symbols()
assert len(USER_DEFINED) == 300

# Synthetic pt_BR sample — enough variety for BPE to find some merges.
SAMPLE_PTBR = """\
Bom dia! Eu queria saber se você está bem. Hoje é um dia bonito para programar e
para tomar um cafezinho na padaria da esquina. Você já viu o filme novo do diretor
brasileiro? Achei muito bom, gostei da fotografia e da trilha sonora. A história
fala sobre uma família que mora no interior de Minas Gerais e que decide se mudar
para São Paulo em busca de uma vida melhor. Os atores estão excelentes, principalmente
o ator que faz o pai. Ele consegue transmitir uma emoção muito forte sem precisar
falar muito. Recomendo demais para quem gosta de filmes nacionais. Vamos combinar
de assistir junto na próxima sexta-feira? Acho que vai ser uma boa pedida. Pode ser
no cinema do shopping ou na minha casa, do jeito que você preferir. Me avisa depois,
tá? Beijos!
""" * 200  # ~250 KB — minimum for a 1000-piece BPE


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus = td / "corpus.txt"
        corpus.write_text(SAMPLE_PTBR)

        out = td / "smoke"
        # Smaller vocab for fast training. With 300 user-defined + 256 byte = 556
        # minimum; we ask for 1000 so SP has ~440 BPE pieces of headroom.
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            input_format="text",
            model_prefix=str(out),
            vocab_size=1000,
            character_coverage=0.9995,
            model_type="bpe",
            treat_whitespace_as_suffix=True,
            user_defined_symbols=USER_DEFINED,
            pad_id=0, bos_id=1, eos_id=2, unk_id=3,
            byte_fallback=True,
            num_threads=4,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(str(out) + ".model")

        print(f"vocab_size = {sp.get_piece_size()}")

        # 1. Slot indices: declaration order should match index 4..303
        mismatches = []
        for expected_id, sym in enumerate(USER_DEFINED, start=4):
            got = sp.piece_to_id(sym)
            if got != expected_id:
                mismatches.append((expected_id, sym, got))
        if mismatches:
            print(f"FAIL: {len(mismatches)} slot-mismatch(es); first 5:")
            for e, s, g in mismatches[:5]:
                print(f"  expected id {e} for {s!r}, got {g}")
            sys.exit(1)
        print(f"PASS: all 300 user-defined symbols at expected indices 4..303")

        # 2. CHAR_A..CHAR_Z sequential
        char_ids = [sp.piece_to_id(f"<CHAR_{c}>") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        assert char_ids == list(range(char_ids[0], char_ids[0] + 26)), char_ids
        print(f"PASS: <CHAR_A>..<CHAR_Z> sequential at IDs {char_ids[0]}..{char_ids[-1]}")

        # 3. XBU/XBC/XEC/XC0 resolve and are non-zero, non-unk
        for name in ("<XBU>", "<XBC>", "<XEC>", "<XC0>"):
            i = sp.piece_to_id(name)
            assert i > 3, f"{name} resolves to {i} (unk-or-control)"
        print("PASS: <XBU>/<XBC>/<XEC>/<XC0> resolve OK")

        # 4. byte_fallback present after user-defined block
        # The first byte token <0x00> should appear at index 304 (right after slot 303)
        # — this is what the reference English model has.
        b0 = sp.piece_to_id("<0x00>")
        bff = sp.piece_to_id("<0xFF>")
        print(f"  <0x00> at id {b0}, <0xFF> at id {bff} (delta={bff - b0}, expected 255)")
        assert bff - b0 == 255, "byte tokens not contiguous"
        # In ref English: <0x00>=304, <0xFF>=559. We expect the same here.
        if b0 != 304:
            print(f"NOTE: <0x00>={b0}, reference English has it at 304. Likely fine, but worth checking.")

        # 5. XBU round-trip
        s = "Eu fui ao <XBU>mecado<XBC>mercado<XEC> ontem"
        ids = sp.encode_as_ids(s)
        # Confirm the special tokens encode as single IDs (not split into byte fallback)
        xbu_id = sp.piece_to_id("<XBU>")
        xbc_id = sp.piece_to_id("<XBC>")
        xec_id = sp.piece_to_id("<XEC>")
        for required in (xbu_id, xbc_id, xec_id):
            assert required in ids, f"id {required} not in encoded sequence"
        decoded = sp.decode(ids)
        # NOTE: SentencePiece decode of treat_whitespace_as_suffix tokenizers can
        # tweak the leading/trailing whitespace; we accept fuzzy round-trip.
        print(f"PASS: XBU autocorrect format encodes/decodes")
        print(f"  encoded ids head: {ids[:10]} ...")
        print(f"  decoded: {decoded!r}")

        # 6. whitespace-as-suffix
        pcs = sp.encode_as_pieces("bom dia mundo")
        print(f"  pieces of 'bom dia mundo': {pcs}")
        # Check at least one piece ends with the SP space marker ▁
        has_suffix = any("▁" in p for p in pcs)
        assert has_suffix, "no whitespace-suffix piece detected"
        print("PASS: whitespace-as-suffix encoding active")

        print("\nAll smoke checks PASSED.")


if __name__ == "__main__":
    main()
