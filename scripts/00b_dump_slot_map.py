"""Annotated slot-map dump for the 300 user-defined symbols (indices 4..303)."""
from pathlib import Path
from gguf import GGUFReader

ROOT = Path(__file__).resolve().parent.parent
r = GGUFReader(str(ROOT / "reference_model" / "ml4_1_f16_meta_fixed.gguf"))
toks_f = r.fields["tokenizer.ggml.tokens"]
toks = [bytes(toks_f.parts[i]).decode("utf-8", errors="replace") for i in toks_f.data]

sections = [
    (4, 27, "FUTO reserved/filler slots (likely unused — keep as <FUTO0>..<FUTO23>)"),
    (28, 173, "English contractions / common words / bigrams — REPLACE with pt_BR equivalents"),
    (174, 176, "STRUCTURAL: <XBU>/<XBC>/<XEC> — autocorrect format markers (KEEP IDENTICAL)"),
    (177, 181, "STRUCTURAL: <XC0>..<XC4> — swipe-typing markers (KEEP IDENTICAL)"),
    (182, 207, "STRUCTURAL: <CHAR_A>..<CHAR_Z> — per-key tokens (KEEP IDENTICAL)"),
    (208, 263, "More common English words/bigrams — REPLACE with pt_BR equivalents"),
    (264, 303, "Emoji — can keep or curate for pt_BR audience (40 slots)"),
]

out = ROOT / "notes" / "reference_slot_map.md"
with open(out, "w") as f:
    f.write("# Reference English FUTO model — user-defined symbol slot map\n\n")
    f.write("Source: `tokenizer.ggml.tokens`, indices 4..303 (300 slots, type=USER_DEFINED).\n")
    f.write("These are the structural / special tokens. The pt_BR model must preserve\n")
    f.write("the **structural** slot indices (174..207: XBU/XBC/XEC/XC0-4/CHAR_A-Z) and may\n")
    f.write("replace the **content** slots (contractions, common words, emoji) with pt_BR equivalents.\n")
    f.write("Whether the keyboard app indexes structural tokens by *name lookup* in\n")
    f.write("`tokenizer.ggml.tokens` or by *fixed integer ID* is unclear — preserving the same\n")
    f.write("indices is the safe choice and what this plan does.\n\n")
    for start, end, label in sections:
        f.write(f"## Indices {start}..{end}: {label}\n\n")
        f.write("```\n")
        for i in range(start, end + 1):
            f.write(f"{i:4d}  {toks[i]!r}\n")
        f.write("```\n\n")

print(f"wrote {out} ({out.stat().st_size} bytes)")
