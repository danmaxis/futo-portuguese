"""
Phase 5: end-to-end GGUF assembly for the FUTO Keyboard pt_BR model.

Pipeline:
  1. Stage the HF checkpoint into a temp dir, copying spm_pt_br.model
     into it as `tokenizer.model` (Llama convention) plus the required
     tokenizer_config.json / special_tokens_map.json so that
     convert_hf_to_gguf.py recognises it.
  2. Run llama.cpp/convert_hf_to_gguf.py on the staged dir → vanilla GGUF.
  3. Run 06_patch_metadata.py on the vanilla GGUF → final FUTO-flavoured GGUF
     with keyboardlm.* fields.
  4. Emit summary diff vs the reference English model's metadata.

Usage:
  python 05_to_futo_gguf.py \\
      --checkpoint finetune/final \\
      --tokenizer tokenizer/spm_pt_br.model \\
      --llama-cpp /path/to/llama.cpp \\
      --out gguf/pt_br_futo.gguf

Designed to run on this VM (CPU; no GPU needed). Pulls the checkpoint from the
GPU host with rsync first if --checkpoint points at a remote URL like
gpu-train:/workspace/finetune/final.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SPECIAL_TOKENS_MAP = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}

TOKENIZER_CONFIG = {
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 2048,
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "add_bos_token": True,
    "add_eos_token": False,
    "clean_up_tokenization_spaces": False,
    "legacy": False,
}


def stage_checkpoint(checkpoint: Path, sp_model: Path, dest: Path) -> None:
    """Copy the model checkpoint files + the SP tokenizer into a single staging dir."""
    dest.mkdir(parents=True, exist_ok=True)
    # Copy model files
    copied = 0
    for f in checkpoint.iterdir():
        if f.name in {"tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
            continue  # we'll write fresh ones
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            copied += 1
    print(f"  Copied {copied} checkpoint files")
    # SP tokenizer → tokenizer.model (Llama convention)
    shutil.copy2(sp_model, dest / "tokenizer.model")
    print(f"  Copied {sp_model.name} → tokenizer.model")
    # Tokenizer config / special tokens map
    (dest / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2))
    (dest / "special_tokens_map.json").write_text(json.dumps(SPECIAL_TOKENS_MAP, indent=2))
    print("  Wrote tokenizer_config.json + special_tokens_map.json")


def run_convert(llama_cpp: Path, staged: Path, out_vanilla: Path) -> None:
    cmd = [
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(staged),
        "--outfile", str(out_vanilla),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_patch(scripts_dir: Path, in_gguf: Path, out_gguf: Path, sp_model: Path,
              languages: str, features: str, history: str | None) -> None:
    cmd = [
        sys.executable,
        str(scripts_dir / "06_patch_metadata.py"),
        "--in", str(in_gguf),
        "--out", str(out_gguf),
        "--tokenizer", str(sp_model),
        "--languages", languages,
        "--features", features,
    ]
    if history:
        cmd += ["--history", history]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def diff_metadata(llama_cpp: Path, vanilla: Path, ours: Path, reference: Path, out_diff: Path) -> None:
    dump = llama_cpp / "gguf-py" / "gguf" / "scripts" / "gguf_dump.py"

    def dump_one(path: Path, target: Path) -> None:
        with open(target, "w") as f:
            subprocess.run([sys.executable, str(dump), str(path)], stdout=f, check=True)

    notes = ours.parent.parent / "notes"
    notes.mkdir(exist_ok=True)
    ours_dump = notes / "our_metadata.txt"
    dump_one(ours, ours_dump)
    print(f"  Wrote {ours_dump}")

    if reference.exists():
        result = subprocess.run(
            ["diff", str(reference), str(ours_dump)],
            capture_output=True, text=True
        )
        out_diff.write_text(result.stdout)
        # show only the diff highlights
        added = sum(1 for line in result.stdout.splitlines() if line.startswith("> "))
        removed = sum(1 for line in result.stdout.splitlines() if line.startswith("< "))
        print(f"  Diff vs reference: {removed} lines unique to reference, {added} lines unique to ours")
        print(f"  Full diff: {out_diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir (finetune/final/)")
    ap.add_argument("--tokenizer", required=True, help="spm_pt_br.model")
    ap.add_argument("--llama-cpp", required=True, help="Path to local llama.cpp clone")
    ap.add_argument("--out", required=True, help="Final GGUF output path (e.g. gguf/pt_br_futo.gguf)")
    ap.add_argument("--reference", default="reference_model/ml4_1_f16_meta_fixed.gguf",
                    help="Reference English model for diff (optional)")
    ap.add_argument("--languages", default="pt-BR")
    ap.add_argument("--features", default="base_v1 inverted_space xbu_char_autocorrect_v1")
    ap.add_argument("--history", default=None)
    ap.add_argument("--keep-staged", action="store_true",
                    help="Keep the staged checkpoint dir for debugging")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    sp_model = Path(args.tokenizer)
    llama_cpp = Path(args.llama_cpp)
    out = Path(args.out)
    reference = Path(args.reference)
    scripts_dir = Path(__file__).resolve().parent

    if not checkpoint.exists():
        sys.exit(f"Checkpoint dir not found: {checkpoint}")
    if not sp_model.exists():
        sys.exit(f"SentencePiece model not found: {sp_model}")
    if not (llama_cpp / "convert_hf_to_gguf.py").exists():
        sys.exit(f"convert_hf_to_gguf.py not found in {llama_cpp}")

    out.parent.mkdir(parents=True, exist_ok=True)
    notes_dir = out.parent.parent / "notes"
    notes_dir.mkdir(exist_ok=True)

    # 1. Stage
    if args.keep_staged:
        staged = out.parent / "_staged"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        ctx = None
    else:
        ctx = tempfile.TemporaryDirectory()
        staged = Path(ctx.name)

    print(f"[1/3] Staging checkpoint into {staged}")
    stage_checkpoint(checkpoint, sp_model, staged)

    # 2. Convert HF → GGUF
    vanilla = out.with_suffix(".vanilla.gguf")
    print(f"[2/3] HF -> vanilla GGUF: {vanilla}")
    run_convert(llama_cpp, staged, vanilla)
    print(f"  vanilla GGUF size: {vanilla.stat().st_size:,} bytes")

    # 3. Patch with FUTO metadata
    print(f"[3/3] Patching with keyboardlm.* fields -> {out}")
    run_patch(scripts_dir, vanilla, out, sp_model, args.languages, args.features, args.history)
    print(f"  final GGUF size: {out.stat().st_size:,} bytes")

    # 4. Diff against reference
    print("[4/4] Diffing against reference English model")
    diff_metadata(llama_cpp, vanilla, out, reference, notes_dir / "metadata_diff.txt")

    if ctx is not None:
        ctx.cleanup()
    else:
        print(f"  Staged dir kept at: {staged}")

    print()
    print(f"DONE: {out}")
    print()
    print("Next steps:")
    print(f"  1. Inspect: python {llama_cpp}/gguf-py/gguf/scripts/gguf_dump.py {out} | head -40")
    print(f"  2. Inference smoke test: {llama_cpp}/build/bin/llama-cli -m {out} -p 'Bom dia <XBU>obigado<XBC>' -n 10")
    print(f"  3. Transfer to phone and side-load via FUTO Keyboard's Languages & Models import.")


if __name__ == "__main__":
    main()
