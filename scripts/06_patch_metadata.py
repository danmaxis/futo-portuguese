"""
Phase 5 (final step): patch a fresh GGUF with FUTO-required metadata fields.

After `convert_hf_to_gguf.py` produces a vanilla Llama GGUF, this script:
  1. Copies all original tensors and standard fields
  2. Overwrites general.name to a pt_BR-friendly label
  3. Adds the keyboardlm.* fields the FUTO Android app validates (verified
     against the reference English model's gguf_dump):
       keyboardlm.languages           STRING  'pt-BR'
       keyboardlm.finetuning_count    UINT32  0
       keyboardlm.history             STRING  '<date>: Model created (pt-BR v1)'
       keyboardlm.features            STRING  'base_v1 inverted_space xbu_char_autocorrect_v1'
       keyboardlm.ext_tokenizer_type  STRING  'sentencepiece'
       keyboardlm.ext_tokenizer_data  [UINT8] raw bytes of spm_pt_br.model

Usage:
  python 06_patch_metadata.py \\
      --in gguf/pt_br_base.gguf \\
      --out gguf/pt_br_futo.gguf \\
      --tokenizer tokenizer/spm_pt_br.model \\
      --languages pt-BR \\
      --features 'base_v1 inverted_space xbu_char_autocorrect_v1' \\
      --history '2026-04-28: Model created (pt-BR v1)'
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


# Fields we explicitly set (everything else is copied verbatim)
OVERRIDDEN = {
    "general.name",
    "keyboardlm.languages",
    "keyboardlm.finetuning_count",
    "keyboardlm.history",
    "keyboardlm.features",
    "keyboardlm.ext_tokenizer_type",
    "keyboardlm.ext_tokenizer_data",
}

# Fields managed automatically by GGUFWriter — DO NOT copy or it'll duplicate
WRITER_MANAGED = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


def _value(field):
    """Decode a GGUFReader field to a Python value."""
    t = field.types[0]
    if t == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if len(field.types) == 1:
        # scalar
        p = field.parts[field.data[0]]
        return p.tolist()[0] if hasattr(p, "tolist") else p
    # array
    inner = field.types[1]
    if inner == GGUFValueType.STRING:
        return [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i]
            for i in field.data]


def _copy_field(writer: GGUFWriter, name: str, field):
    """Copy a field from a reader to a writer, preserving its type."""
    t = field.types[0]
    val = _value(field)
    if t == GGUFValueType.STRING:
        writer.add_string(name, val)
    elif t == GGUFValueType.UINT32:
        writer.add_uint32(name, val)
    elif t == GGUFValueType.INT32:
        writer.add_int32(name, val)
    elif t == GGUFValueType.UINT64:
        writer.add_uint64(name, val)
    elif t == GGUFValueType.INT64:
        writer.add_int64(name, val)
    elif t == GGUFValueType.FLOAT32:
        writer.add_float32(name, val)
    elif t == GGUFValueType.FLOAT64:
        writer.add_float64(name, val)
    elif t == GGUFValueType.BOOL:
        writer.add_bool(name, val)
    elif t == GGUFValueType.UINT8:
        writer.add_uint8(name, val)
    elif t == GGUFValueType.INT8:
        writer.add_int8(name, val)
    elif t == GGUFValueType.UINT16:
        writer.add_uint16(name, val)
    elif t == GGUFValueType.INT16:
        writer.add_int16(name, val)
    elif t == GGUFValueType.ARRAY:
        # Mixed-type array — use the inner type
        inner = field.types[1]
        if inner == GGUFValueType.STRING:
            writer.add_array(name, val)
        else:
            # Numeric array — gguf-py's add_array handles lists; verify by smoke
            writer.add_array(name, val)
    else:
        raise ValueError(f"Unhandled GGUF type {t} for field {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input GGUF (from convert_hf_to_gguf.py)")
    ap.add_argument("--out", required=True, help="Output GGUF with FUTO metadata")
    ap.add_argument("--tokenizer", required=True, help="Path to spm_pt_br.model (raw bytes embedded)")
    ap.add_argument("--languages", default="pt-BR")
    ap.add_argument("--features", default="base_v1 inverted_space xbu_char_autocorrect_v1")
    ap.add_argument("--history", default=None,
                    help="Defaults to today's date + 'Model created (pt-BR v1)'")
    ap.add_argument("--name", default="Brazilian Portuguese v1")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    spm_path = Path(args.tokenizer)
    if not in_path.exists():
        raise SystemExit(f"Input GGUF not found: {in_path}")
    if not spm_path.exists():
        raise SystemExit(f"SentencePiece model not found: {spm_path}")

    history = args.history or f"{datetime.date.today().isoformat()}: Model created (pt-BR v1)"

    print(f"Reading {in_path}")
    reader = GGUFReader(str(in_path))

    # Identify the architecture
    arch_field = reader.fields["general.architecture"]
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    print(f"Architecture: {arch}")

    print(f"Writing {out_path}")
    writer = GGUFWriter(str(out_path), arch)

    # 1. Copy all original fields except those we override or that GGUFWriter manages
    skipped_count = 0
    copied_count = 0
    for name, field in reader.fields.items():
        if name in WRITER_MANAGED:
            continue
        if name in OVERRIDDEN:
            skipped_count += 1
            continue
        _copy_field(writer, name, field)
        copied_count += 1
    print(f"Copied {copied_count} fields, override {skipped_count + len(OVERRIDDEN)}, writer-managed {len(WRITER_MANAGED)}")

    # 2. Set FUTO-required keyboardlm.* fields and general.name
    spm_bytes = spm_path.read_bytes()
    print(f"Embedded SentencePiece: {len(spm_bytes):,} bytes")
    writer.add_string("general.name", args.name)
    writer.add_string("keyboardlm.languages", args.languages)
    writer.add_uint32("keyboardlm.finetuning_count", 0)
    writer.add_string("keyboardlm.history", history)
    writer.add_string("keyboardlm.features", args.features)
    writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    # MUST be a UINT8 array (matches reference English model). The high-level
    # add_array auto-infers INT32 from int elements; we use add_key_value
    # directly to force sub_type=UINT8. `bytes` satisfies the Sequence check.
    writer.add_key_value(
        "keyboardlm.ext_tokenizer_data",
        spm_bytes,
        GGUFValueType.ARRAY,
        sub_type=GGUFValueType.UINT8,
    )

    # 3. Copy tensors unchanged (preserves quantization)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    print(f"Copied {len(reader.tensors)} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print()
    print("Next: dump and diff against reference")
    print(f"  python llama.cpp/gguf-py/gguf/scripts/gguf_dump.py {out_path} > notes/our_metadata.txt")
    print(f"  diff notes/reference_metadata.txt notes/our_metadata.txt")


if __name__ == "__main__":
    main()
