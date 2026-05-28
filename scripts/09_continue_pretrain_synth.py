"""
Continue-pretrain a checkpoint on a fresh synth corpus (Wu et al. 2024 recipe).

Loads an existing LlamaForCausalLM checkpoint (e.g. pretrain_big/checkpoint-100000)
and continues training on a small, high-diversity corpus of LLM-generated pt-BR
chat-style text. Output: a sharper base for downstream stage_a/b/c finetuning.

This is intentionally short and minimal — no new model architecture, no XBU
format, just clean LM loss on text shards. Designed for the 5070 Ti (16GB) but
runs anywhere.

Usage:
  python3 scripts/09_continue_pretrain_synth.py \\
      --base pretrain_big/checkpoint-100000 \\
      --tokenizer tokenizer/spm_pt_br_v2.model \\
      --corpus corpora/synth_v82 \\
      --out pretrain_big_v82 \\
      --total-steps 5000 \\
      --micro-batch 16 --grad-accum 8 \\
      --seq-len 512 \\
      --lr 1e-4 --warmup 200
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path

import torch
import sentencepiece as spm
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, set_seed

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_progress import ProgressCallback
from torch.utils.data import IterableDataset

# Reuse the streamer from 03_pretrain to keep packing identical.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_pretrain_mod", str(Path(__file__).resolve().parent / "03_pretrain.py"))
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
PtBrShardStreamer = _pre.PtBrShardStreamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base checkpoint (e.g. pretrain_big/checkpoint-100000)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True, help="Directory of shard_*.txt files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Lower than fresh pretrain — we're refining, not starting over.")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    shards = sorted(glob.glob(str(Path(args.corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards found in {args.corpus}")
    print(f"Found {len(shards)} corpus shards")

    train_ds = PtBrShardStreamer(
        shard_paths=shards,
        sp_model_path=args.tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    print(f"Loading base: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n/1e6:.2f}M")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=args.total_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=50,
        save_steps=args.save_every,
        save_total_limit=5,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=args.seed,
        disable_tqdm=False,
    )

    progress_log = str(out / "progress.log")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        callbacks=[ProgressCallback(phase="continue-pretrain", seq_len=args.seq_len,
                                    log_path=progress_log)],
    )

    print(f"Continue-pretrain: {args.total_steps} steps, "
          f"global batch = {args.micro_batch}*{args.grad_accum} = "
          f"{args.micro_batch*args.grad_accum}, seq_len={args.seq_len}, lr={args.lr}")
    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved to {out}/final/")


if __name__ == "__main__":
    main()
