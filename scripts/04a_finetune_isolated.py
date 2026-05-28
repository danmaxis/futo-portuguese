"""
Phase 4a: Isolated <XBU>typo<XBC>correct<XEC> fine-tune.

Streams pairs from pre-generated JSONL files (synthetic + real user typos),
formats each pair as `<XBU><CHAR_*>...<XBC>correct<XEC>`, and continues training
from the base checkpoint.

Loss formulation: weighted cross-entropy via Prompt Loss Weight (PLW).
  --plw 0.0  (default)  full mask on non-correction tokens (the historical behavior)
  --plw 0.05            relaxed mask per arxiv 2401.13586 — small positive weight
                        on clean tokens, may improve over PLW=0 for short completions

Eval callback runs against `--eval-jsonl` every `--eval-every` steps and writes
top-1/top-5 trajectory to a CSV alongside the checkpoint.
"""
from __future__ import annotations
import argparse
import os
import random
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_typo_synthesis import make_xbu_triple
from lib_progress import ProgressCallback
from lib_plw_trainer import PLWTrainer, SAMPLWTrainer, build_loss_weights_for_correction_only
from lib_real_eval_callback import RealTypoEvalCallback


class JsonlTriplesDataset(IterableDataset):
    """
    Streams <XBU>typo<XBC>correct<XEC> examples from pre-generated JSONL files.

    Mix policy: with probability `real_mix_ratio` draw from real_jsonl,
    otherwise from synth_jsonl. Each example is one triple, BOS-prefixed,
    padded to seq_len. Emits `loss_weights` for PLW training.
    """
    def __init__(self, synth_jsonl: str, real_jsonl: str,
                 sp_model_path: str, seq_len: int = 64,
                 real_mix_ratio: float = 0.25, plw_clean: float = 0.0,
                 seed: int = 1337):
        import json
        self.sp_model_path = sp_model_path
        self.seq_len = seq_len
        self.seed = seed
        self.real_mix_ratio = real_mix_ratio
        self.plw_clean = plw_clean
        self.synth_pairs = json.loads(Path(synth_jsonl).read_text())
        self.real_pairs = json.loads(Path(real_jsonl).read_text())
        print(f"[dataset] synth={len(self.synth_pairs)} real={len(self.real_pairs)} "
              f"real_mix_ratio={real_mix_ratio:.2f} plw_clean={plw_clean}")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + worker_id * 9973)

        sp = spm.SentencePieceProcessor()
        sp.load(self.sp_model_path)
        bos = sp.bos_id()
        eos = sp.eos_id()
        pad = sp.pad_id()
        xbc_id = sp.piece_to_id("<XBC>")
        xec_id = sp.piece_to_id("<XEC>")

        while True:
            if rng.random() < self.real_mix_ratio and self.real_pairs:
                pair = rng.choice(self.real_pairs)
            else:
                pair = rng.choice(self.synth_pairs)
            typo, correct = pair["typed"], pair["committed"]
            if not typo or not correct or typo == correct:
                continue
            triple = make_xbu_triple(typo, correct)
            ids = [bos] + sp.encode(triple, out_type=int) + [eos]
            if len(ids) > self.seq_len:
                continue
            input_ids = ids + [pad] * (self.seq_len - len(ids))
            # Labels: same shape as input_ids (HF Llama shifts internally).
            # Padding positions get -100 so they don't contribute to denominator.
            labels = [t if t != pad else -100 for t in input_ids]
            # Per-token weights: 1.0 for correction span (XBC..XEC), plw_clean for
            # everything else (BOS, XBU, CHAR_*). PLW=0 ≡ historical full-mask.
            loss_weights = build_loss_weights_for_correction_only(
                input_ids, xbc_id=xbc_id, xec_id=xec_id,
                plw_clean=self.plw_clean, in_span_weight=1.0,
            )
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
                "attention_mask": torch.tensor(
                    [1 if t != pad else 0 for t in input_ids], dtype=torch.long
                ),
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Pretrained base checkpoint dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--synth-jsonl", required=True)
    ap.add_argument("--real-jsonl", required=True)
    ap.add_argument("--real-mix-ratio", type=float, default=0.25)
    ap.add_argument("--plw", type=float, default=0.0,
                    help="Prompt Loss Weight for clean (non-correction) tokens. "
                         "0.0 = full mask (historical 4a behavior). 0.05 = relaxed mask "
                         "per arxiv 2401.13586 — recommended for Phase 1 A2 ablation.")
    ap.add_argument("--out", default="finetune/stage_a")
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--save-total-limit", type=int, default=3,
                    help="HF Trainer save_total_limit (keeps last N checkpoints). "
                         "Set to a large number to keep all intermediate checkpoints.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wandb-project", type=str, default="futo-pt-br")
    ap.add_argument("--progress-log", type=str, default=None)
    ap.add_argument("--eval-jsonl", type=str, default=None,
                    help="Path to notes/real_typos_eval.json. If set, runs real-typo "
                         "eval every --eval-every steps and writes a CSV.")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-sam", action="store_true",
                    help="Enable Sharpness-Aware Minimization (doubles per-step cost). "
                         "Recommended for Phase 1 B2/A3 ablation against forgetting.")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading base checkpoint: {args.base}")
    model = LlamaForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    train_ds = JsonlTriplesDataset(
        synth_jsonl=args.synth_jsonl,
        real_jsonl=args.real_jsonl,
        sp_model_path=args.tokenizer,
        seq_len=args.seq_len,
        real_mix_ratio=args.real_mix_ratio,
        plw_clean=args.plw,
        seed=args.seed,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=args.total_steps,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        save_steps=args.save_every,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["wandb"] if args.wandb_project else [],
        seed=args.seed,
        disable_tqdm=False,
        run_name="phase4a_isolated",
        remove_unused_columns=False,  # keep loss_weights through the pipeline
    )

    progress_log = args.progress_log or str(out / "progress.log")
    Path(progress_log).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [ProgressCallback(phase="stage_a", seq_len=args.seq_len, log_path=progress_log)]
    if args.eval_jsonl:
        callbacks.append(RealTypoEvalCallback(
            eval_jsonl=args.eval_jsonl,
            sp_model_path=args.tokenizer,
            eval_every=args.eval_every,
            csv_path=str(out / "real_typo_eval.csv"),
        ))

    trainer_cls = SAMPLWTrainer if args.use_sam else PLWTrainer
    trainer_kwargs = {"sam_rho": args.sam_rho} if args.use_sam else {}
    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    print(f"Starting Phase 4a: {args.total_steps} steps, plw={args.plw}, "
          f"global batch {args.micro_batch * args.grad_accum}, seq_len {args.seq_len}")
    print(f"Progress log: {progress_log}")
    if args.eval_jsonl:
        print(f"Real-typo eval: every {args.eval_every} steps → {out}/real_typo_eval.csv")

    trainer.train()
    trainer.save_model(str(out / "final"))
    print(f"Saved Phase 4a final checkpoint to {out}/final/")


if __name__ == "__main__":
    main()
