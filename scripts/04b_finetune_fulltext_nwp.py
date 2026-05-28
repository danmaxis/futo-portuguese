"""
Phase 3 ablation: Plain NWP fine-tune (NO XBU/XBC/XEC format).

This is identical to 04b_finetune_fulltext.py EXCEPT it does NOT call
`make_inline_corrected` — it just streams raw corpus text and trains
straightforward next-word prediction. The hypothesis (per SwiftKey's 2025
production paper) is that FUTO's higher-level decoder might produce good
autocorrect on-device even without the XBU format, in which case we'd drop
the format entirely.

Loss formulation: PLW=0.05 same as 04b. Eval callback wired same way.

Side-by-side test: train this AND the XBU twin from the same base. Package
both as GGUF, smoke-test on phone. Whichever produces better on-device
autocorrect wins.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm
from transformers import (
    LlamaForCausalLM,
    TrainingArguments,
    set_seed,
)
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_progress import ProgressCallback
from lib_plw_trainer import PLWTrainer, SAMPLWTrainer
from lib_real_eval_callback import RealTypoEvalCallback


class PlainNWPDataset(IterableDataset):
    """
    Same as 04b's InlineCorruptedDataset but WITHOUT the XBU wrap. Streams
    corpus shards, tokenizes, packs into fixed-length sequences for plain
    next-word prediction. All tokens contribute equally to loss (PLW=1 implicit
    since there are no XBU spans to mask around).

    Optionally we can still apply PLW < 1 to e.g. lower the weight on punctuation
    or rare tokens — but for the Phase 3 ablation we just do uniform NWP.
    """
    def __init__(self, shard_paths: list[str], sp_model_path: str,
                 seq_len: int = 512, seed: int = 1337,
                 shuffle_buffer: int = 1024):
        self.shard_paths = sorted(shard_paths)
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _iter_shards(self, worker_id: int, num_workers: int):
        for i, path in enumerate(self.shard_paths):
            if i % num_workers != worker_id:
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rng = random.Random(self.seed + worker_id * 9973)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos = sp.bos_id()
        eos = sp.eos_id()

        buffer: list[int] = []
        line_buffer: list[str] = []

        for line in self._iter_shards(worker_id, num_workers):
            line_buffer.append(line)
            if len(line_buffer) >= self.shuffle_buffer:
                rng.shuffle(line_buffer)
                for raw in line_buffer:
                    # NO make_inline_corrected — plain text only
                    buffer.append(bos)
                    buffer.extend(sp.encode(raw, out_type=int))
                    buffer.append(eos)
                    while len(buffer) >= self.seq_len:
                        ids = buffer[: self.seq_len]
                        del buffer[: self.seq_len]
                        # Uniform weight on all tokens. Same shape convention
                        # as 04b: input_ids == labels, HF shifts internally.
                        yield {
                            "input_ids": torch.tensor(ids, dtype=torch.long),
                            "labels": torch.tensor(ids, dtype=torch.long),
                            "loss_weights": torch.ones(len(ids), dtype=torch.float32),
                        }
                line_buffer.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="finetune/stage_b_nwp")
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--progress-log", type=str, default=None)
    ap.add_argument("--eval-jsonl", type=str, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-sam", action="store_true")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading base checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    shards = sorted(glob.glob(str(Path(args.corpus) / "shard_*.txt")))
    if not shards:
        raise SystemExit(f"No shards in {args.corpus}")
    print(f"Found {len(shards)} shards (plain NWP, no XBU wrap)")

    train_ds = PlainNWPDataset(
        shard_paths=shards, sp_model_path=args.tokenizer,
        seq_len=args.seq_len, seed=args.seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out), max_steps=args.total_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_steps=args.warmup,
        weight_decay=0.01, lr_scheduler_type="cosine",
        bf16=True, logging_steps=50,
        save_steps=args.save_every, save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=args.seed, disable_tqdm=False,
        run_name="phase3_nwp_only",
        remove_unused_columns=False,
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_b_nwp", seq_len=args.seq_len, log_path=progress_log)]
    if args.eval_jsonl:
        callbacks.append(RealTypoEvalCallback(
            eval_jsonl=args.eval_jsonl, sp_model_path=args.tokenizer,
            eval_every=args.eval_every, csv_path=str(out / "real_typo_eval.csv"),
        ))

    trainer_cls = SAMPLWTrainer if args.use_sam else PLWTrainer
    trainer_kwargs = {"sam_rho": args.sam_rho} if args.use_sam else {}
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds,
                          callbacks=callbacks, **trainer_kwargs)

    print(f"Starting Phase 3 NWP fine-tune: {args.total_steps} steps, "
          f"global batch {args.micro_batch * args.grad_accum}, seq_len {args.seq_len}")
    print(f"Progress log: {progress_log}")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved NWP final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
