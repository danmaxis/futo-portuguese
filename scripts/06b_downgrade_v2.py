"""
Downgrade GGUF v3 → v2 + strip fields not present in the reference English
FUTO model. This is a workaround for the FUTO Android app's vendored llama.cpp
which only handles GGUF v2 and chokes on newer fields (e.g.,
`tokenizer.ggml.pre`, `llama.attention.key_length`, etc.).

Diagnosed: same model loads fine via llama-cpp-python on x86_64 (where the
parser is up-to-date), but crashes the keyboard mid-inference on Android.
The reference English FUTO model is GGUF v2 with 28 KV fields.

Usage:
  python 06b_downgrade_v2.py --in gguf/pt_br_futo_mini.gguf --out gguf/pt_br_futo_mini_v2.gguf
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path

from gguf import GGUFReader, GGUFWriter, GGUFValueType


# Fields the FUTO-bundled (older) llama.cpp does NOT understand. Strip them.
# Verified by diffing against `reference_model/ml4_1_f16_meta_fixed.gguf`.
STRIP = {
    "general.size_label",
    "general.type",
    "llama.attention.key_length",
    "llama.attention.value_length",
    "llama.vocab_size",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.pre",
}

# Fields that GGUFWriter manages itself (don't copy)
WRITER_MANAGED = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


def _value(field):
    t = field.types[0]
    if t == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if len(field.types) == 1:
        p = field.parts[field.data[0]]
        return p.tolist()[0] if hasattr(p, "tolist") else p
    inner = field.types[1]
    if inner == GGUFValueType.STRING:
        return [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i]
            for i in field.data]


def _copy_field(writer: GGUFWriter, name: str, field):
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
        # Preserve sub-type; for the keyboardlm.ext_tokenizer_data UINT8 array
        # we need bytes() not list. For other arrays gguf-py infers from elements.
        inner = field.types[1] if len(field.types) > 1 else None
        if inner == GGUFValueType.UINT8:
            # Reconstruct bytes for proper sub-type
            raw = bytes(int(field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist")
                            else field.parts[i]) for i in field.data)
            writer.add_key_value(name, raw, GGUFValueType.ARRAY,
                                 sub_type=GGUFValueType.UINT8)
        else:
            writer.add_array(name, val)
    else:
        raise ValueError(f"Unhandled type {t} for {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)

    print(f"Reading {in_path}")
    reader = GGUFReader(str(in_path))
    arch = bytes(reader.fields["general.architecture"].parts[
        reader.fields["general.architecture"].data[0]]).decode()

    print(f"Writing v2 GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), arch)

    copied = stripped = 0
    for name, field in reader.fields.items():
        if name in WRITER_MANAGED:
            continue
        if name in STRIP:
            stripped += 1
            print(f"  STRIP   {name}")
            continue
        _copy_field(writer, name, field)
        copied += 1
    print(f"Copied {copied}, stripped {stripped}")

    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    print(f"Copied {len(reader.tensors)} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Patch GGUF.version (uint32 LE at offset 4) from 3 → 2
    print(f"Patching GGUF.version 3 → 2 in {out_path}")
    with open(out_path, "r+b") as f:
        f.seek(0)
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic!r})")
        f.seek(4)
        ver = struct.unpack("<I", f.read(4))[0]
        print(f"  current version: {ver}")
        if ver != 3:
            print(f"  WARNING: expected v3, got v{ver}; not touching")
        else:
            f.seek(4)
            f.write(struct.pack("<I", 2))
            print("  written: 2")

    print(f"Done: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
