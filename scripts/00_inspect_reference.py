"""
Deep-introspection of the reference English FUTO model.
Extracts the full keyboardlm.features string, all special tokens, and the
embedded SentencePiece model so we can reproduce the format exactly for pt_BR.

Outputs:
  notes/reference_full_features.txt    full keyboardlm.features string
  notes/reference_special_tokens.txt   list of all non-NORMAL tokens (types 2, 3, 6)
  notes/reference_first_64_tokens.txt  first 64 tokens (indices 0..63) including <FUTO*>
  reference_model/extracted_spm.model  the embedded SentencePiece .model bytes
  notes/reference_spm_tokens.txt       SentencePiece vocab dump (after extraction)
"""
import os
import sys
from pathlib import Path
from gguf import GGUFReader, GGUFValueType

ROOT = Path(__file__).resolve().parent.parent
GGUF_PATH = ROOT / "reference_model" / "ml4_1_f16_meta_fixed.gguf"
NOTES = ROOT / "notes"
NOTES.mkdir(exist_ok=True)

reader = GGUFReader(str(GGUF_PATH))

def get_field_value(field):
    """Return a Python value for a gguf field, handling strings, scalars, and arrays."""
    if field.types[0] == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if len(field.types) == 1:
        # scalar
        return field.parts[field.data[0]].tolist()[0] if hasattr(field.parts[field.data[0]], "tolist") else field.parts[field.data[0]]
    # array
    if field.types[1] == GGUFValueType.STRING:
        return [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]
    return [field.parts[idx].tolist()[0] if hasattr(field.parts[idx], "tolist") else field.parts[idx] for idx in field.data]


print("=" * 60)
print("Loaded:", GGUF_PATH)
print("Fields:", len(reader.fields))
print()

# 1. Dump keyboardlm.* fields fully
kbd_fields = {k: v for k, v in reader.fields.items() if k.startswith("keyboardlm")}
print("keyboardlm.* fields:")
for k, f in kbd_fields.items():
    if k == "keyboardlm.ext_tokenizer_data":
        size = len(f.data)
        print(f"  {k}: <{size} bytes of embedded SentencePiece model>")
        spm_bytes = bytes([f.parts[i].tolist()[0] if hasattr(f.parts[i], "tolist") else f.parts[i] for i in f.data])
        out = ROOT / "reference_model" / "extracted_spm.model"
        out.write_bytes(spm_bytes)
        print(f"    -> wrote {out} ({out.stat().st_size} bytes)")
    else:
        v = get_field_value(f)
        print(f"  {k}: {v!r}")

# 2. Save full features string
features_field = reader.fields.get("keyboardlm.features")
if features_field:
    features = get_field_value(features_field)
    (NOTES / "reference_full_features.txt").write_text(features + "\n")
    print(f"\nFull keyboardlm.features:\n  {features}\n")

# 3. Tokens — find non-NORMAL tokens (types: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE)
tokens_field = reader.fields["tokenizer.ggml.tokens"]
types_field = reader.fields["tokenizer.ggml.token_type"]

tokens = [bytes(tokens_field.parts[idx]).decode("utf-8", errors="replace") for idx in tokens_field.data]
types = [types_field.parts[idx].tolist()[0] if hasattr(types_field.parts[idx], "tolist") else int(types_field.parts[idx]) for idx in types_field.data]

print(f"Total tokens: {len(tokens)}")
type_counts = {}
for t in types:
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"Token type counts: {type_counts}")

# First 64 tokens (special-token range, includes pad/bos/eos/unk and <FUTO*>)
first_64 = "\n".join(f"{i:4d}  type={types[i]}  {tokens[i]!r}" for i in range(min(64, len(tokens))))
(NOTES / "reference_first_64_tokens.txt").write_text(first_64 + "\n")
print("\nFirst 64 tokens written to notes/reference_first_64_tokens.txt")
print("Sample (first 16):")
for i in range(16):
    print(f"  {i:3d}  type={types[i]}  {tokens[i]!r}")

# All non-NORMAL tokens (types != 1) — these are the structural ones we must reproduce
non_normal = [(i, types[i], tokens[i]) for i in range(len(tokens)) if types[i] != 1]
lines = [f"{i:5d}  type={t}  {tok!r}" for i, t, tok in non_normal]
(NOTES / "reference_special_tokens.txt").write_text("\n".join(lines) + "\n")
print(f"\nNon-NORMAL tokens: {len(non_normal)} (saved to notes/reference_special_tokens.txt)")
print("First 30 non-NORMAL:")
for i, t, tok in non_normal[:30]:
    print(f"  {i:4d}  type={t}  {tok!r}")

# 4. llama.* config
print("\nllama.* config (for our pt_BR LlamaConfig):")
for k, f in sorted(reader.fields.items()):
    if k.startswith("llama."):
        print(f"  {k}: {get_field_value(f)!r}")
