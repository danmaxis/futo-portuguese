"""
Compact progress logging for HF Trainer-based scripts (Phase 3, 4a, 4b, 4c).

Each `[progress]` line is one self-contained snapshot of the run, designed for
one-shot status checks via `tail -1 <log>` or streaming via `grep '\\[progress\\]'`.

Format:
  [start]    t=<utc> phase=<name> total_steps=<N> seq_len=<S> global_batch=<B>
  [progress] t=<utc> step=<i>/<N> pct=<x.x> elapsed=<sec> eta=<sec>
             loss=<f> lr=<f> tps=<int> gpu_mem_gb=<f>
  [checkpoint] t=<utc> step=<i> path=<dir>
  [done]     t=<utc> step=<i>/<N> elapsed=<sec> final_loss=<f> path=<dir>
  [error]    t=<utc> step=<i> msg=<short>

All pairs are space-separated `key=value`, no commas, no quotes — trivially
parseable with `awk '{split($0, kv, " ")...}` or Python `dict(p.split("=",1) for p in line.split())`.
"""
from __future__ import annotations
import datetime
import math
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_kv(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def _emit(stream, kind: str, **kv):
    line = f"[{kind}] t={_utc()} " + _fmt_kv(kv)
    print(line, file=stream, flush=True)


class ProgressCallback(TrainerCallback):
    """
    Emits compact `[progress]` lines for parseable run-status reporting.

    Args:
        phase: short label, e.g. "pretrain", "stage_a", "stage_b".
        seq_len: training sequence length, used for tokens/sec computation.
        log_path: optional file to mirror the lines to (in addition to stdout).
    """
    def __init__(self, phase: str, seq_len: int, log_path: str | None = None):
        self.phase = phase
        self.seq_len = seq_len
        self.log_path = Path(log_path) if log_path else None
        self.log_file = open(self.log_path, "a") if self.log_path else None
        self.start_time: float | None = None
        self.tokens_per_step: int = 0
        self.last_loss: float = 0.0
        self.last_step: int = 0

    def _emit_both(self, kind: str, **kv):
        _emit(sys.stdout, kind, **kv)
        if self.log_file:
            _emit(self.log_file, kind, **kv)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        global_batch = (
            args.per_device_train_batch_size *
            args.gradient_accumulation_steps *
            max(1, args.world_size if hasattr(args, "world_size") else 1)
        )
        self.tokens_per_step = global_batch * self.seq_len
        self._emit_both(
            "start",
            phase=self.phase,
            total_steps=args.max_steps,
            seq_len=self.seq_len,
            global_batch=global_batch,
            tokens_per_step=self.tokens_per_step,
            warmup=args.warmup_steps,
            lr=args.learning_rate,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Skip the "train_runtime" final log — handled in on_train_end.
        if "train_runtime" in logs:
            return
        if state.global_step == 0:
            return
        elapsed = time.time() - self.start_time
        steps_done = state.global_step
        steps_left = max(0, args.max_steps - steps_done)
        # Use steps-per-second from this run, not the most recent interval, for stability
        sps = steps_done / elapsed if elapsed > 0 else 0
        eta = steps_left / sps if sps > 0 else 0
        tokens_done = steps_done * self.tokens_per_step
        tps = int(tokens_done / elapsed) if elapsed > 0 else 0
        gpu_mem_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                      if torch.cuda.is_available() else 0.0)
        loss = logs.get("loss", logs.get("train_loss", float("nan")))
        if not math.isnan(loss):
            self.last_loss = loss
        lr = logs.get("learning_rate", 0)

        self._emit_both(
            "progress",
            step=f"{steps_done}/{args.max_steps}",
            pct=f"{100*steps_done/max(1,args.max_steps):.2f}",
            elapsed=int(elapsed),
            eta=int(eta),
            loss=f"{loss:.4f}" if not math.isnan(loss) else "nan",
            lr=f"{lr:.3e}",
            tps=tps,
            gpu_mem_gb=f"{gpu_mem_gb:.2f}",
        )
        self.last_step = steps_done

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._emit_both("checkpoint", step=state.global_step, path=str(ckpt))

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._emit_both(
            "done",
            phase=self.phase,
            step=f"{state.global_step}/{args.max_steps}",
            elapsed=int(elapsed),
            final_loss=f"{self.last_loss:.4f}",
            path=args.output_dir,
        )
        if self.log_file:
            self.log_file.close()


def status_from_log(log_path: str) -> dict:
    """
    Parse the latest [progress]/[start]/[checkpoint]/[done] events from a log.
    Returns a flat dict with the most recent values of each kind.
    """
    out: dict = {"phase": None, "start": None, "progress": None,
                 "last_checkpoint": None, "done": None, "errors": []}
    try:
        for raw in Path(log_path).read_text(errors="replace").splitlines():
            if not raw.startswith("["):
                continue
            try:
                kind = raw[1:raw.index("]")]
            except ValueError:
                continue
            body = raw[raw.index("]") + 1:].strip()
            kv = {}
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            if kind == "start":
                out["start"] = kv
                out["phase"] = kv.get("phase")
            elif kind == "progress":
                out["progress"] = kv
            elif kind == "checkpoint":
                out["last_checkpoint"] = kv
            elif kind == "done":
                out["done"] = kv
            elif kind == "error":
                out["errors"].append(kv)
    except FileNotFoundError:
        pass
    return out


if __name__ == "__main__":
    # CLI usage: python lib_progress.py /workspace/pretrain.log
    if len(sys.argv) > 1:
        s = status_from_log(sys.argv[1])
        import json
        print(json.dumps(s, indent=2))
