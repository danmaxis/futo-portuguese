"""
Phase 3: pretrain the 36M-param Llama base model on pt_BR.

Architecture matches the reference English FUTO model byte-for-byte:
  - vocab_size=15008, hidden=512, ffn=1024, layers=8, heads=8, head_dim=64
  - max_position=2048, rms_norm_eps=1e-6, rope_theta=10000, MHA (no GQA)
  - tie_word_embeddings=False (output and token_embd are separate tensors)

Designed to run on the RTX 3090 (24 GB) inside the Unraid Docker container.
The 5070 Ti (16 GB) can also run this with reduced micro-batch.

Usage on gpu-train host (3090 container):
  cd /workspace
  source env/bin/activate
  python scripts/03_pretrain.py \\
      --tokenizer tokenizer/spm_pt_br.model \\
      --corpus corpora/clean \\
      --out pretrain \\
      --total-steps 150000 \\
      --micro-batch 16 --grad-accum 16 \\
      --wandb-project futo-pt-br
"""
from __future__ import annotations
import argparse
import glob
import os
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_progress import ProgressCallback


# Verified architecture (notes/reference_metadata.txt + futo_model_schema memory)
def build_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=15008,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,           # MHA, no GQA
        max_position_embeddings=2048,    # llama.context_length from reference
        rms_norm_eps=1e-6,               # NOT 1e-5
        rope_theta=10000.0,
        tie_word_embeddings=False,       # reference has separate output.weight (Q6_K) and token_embd
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n/1e6:.2f}M")
    return model


class PtBrShardStreamer(IterableDataset):
    """
    Streams from `shard_*.txt` files, tokenizes on the fly, packs sequences
    to a fixed length. Worker-aware: each DataLoader worker picks a disjoint
    subset of shards by index modulo, so workers don't read the same data.
    """
    def __init__(self, shard_paths: list[str], sp_model_path: str, seq_len: int = 1024,
                 shuffle_buffer: int = 1024, seed: int = 1337):
        self.shard_paths = sorted(shard_paths)
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_shards(self, worker_id: int, num_workers: int):
        # Each worker gets shards indexed [worker_id::num_workers]
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos, eos = sp.bos_id(), sp.eos_id()

        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for ln in line_buffer:
                    buffer.append(bos)
                    buffer.extend(sp.encode(ln, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len + 1:
                        ids = buffer[: self.seq_len + 1]
                        del buffer[: self.seq_len]
                        # CausalLM: input_ids = ids[:-1], labels = ids[1:]
                        # (HF Trainer handles the shift internally if labels=input_ids)
                        yield {
                            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                            "labels": torch.tensor(ids[1:], dtype=torch.long),
                        }
                line_buffer.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to spm_pt_br.model")
    ap.add_argument("--corpus", required=True, help="Directory of shard_*.txt files")
    ap.add_argument("--out", default="pretrain", help="Output directory for checkpoints")
    ap.add_argument("--total-steps", type=int, default=150_000)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="Training sequence length. <=2048 (architectural max).")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", type=str, default="futo-pt-br")
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--progress-log", type=str, default=None,
                    help="File to mirror compact [progress] lines into (default: <out>/progress.log)")
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

    model = build_model()

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
        gradient_checkpointing=False,    # 36M params + 24GB = no need
        logging_steps=50,
        save_steps=args.save_every,
        save_total_limit=5,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=args.seed,
        disable_tqdm=False,
        # Streaming dataset so eval split is awkward; skip eval during pretrain
        # and rely on perplexity logging plus manual sample-generation in eval script.
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        callbacks=[ProgressCallback(phase="pretrain", seq_len=args.seq_len, log_path=progress_log)],
    )

    print(f"Starting pretrain: {args.total_steps} steps, "
          f"global batch = {args.micro_batch}*{args.grad_accum} = "
          f"{args.micro_batch * args.grad_accum}, seq_len={args.seq_len}")
    print(f"Progress log: {progress_log}")

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(str(out / "base"))
    print(f"Saved final checkpoint to {out}/base/")


if __name__ == "__main__":
    main()
