"""
Trainer callback that evaluates against the held-out real user typo set every
N training steps. Logs top-1 / top-5 to stdout and a CSV so we can plot the
trajectory and localize collapse / improvement steps.

Cost: ~20-30s per eval at defaults (50 pairs, 5 beams, 20 max new tokens).
Negligible vs training time but turns every training run into a debuggable
trajectory rather than a single endpoint datapoint.

Usage in a training script:
    callbacks=[
        ProgressCallback(...),
        RealTypoEvalCallback(
            eval_jsonl="notes/real_typos_eval.json",
            sp_model_path=tokenizer_path,
            eval_every=500,
            csv_path=str(out / "real_typo_eval.csv"),
        ),
    ]
"""
from __future__ import annotations
import json
import unicodedata
from pathlib import Path

import torch
import sentencepiece as spm
from transformers import TrainerCallback


def _to_keypress_chars(typed: str) -> list[str]:
    out = []
    for ch in typed:
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        if base in ("ç", "Ç"):
            base = "C"
        for c in base.upper():
            if "A" <= c <= "Z":
                out.append(f"<CHAR_{c}>")
    return out


class RealTypoEvalCallback(TrainerCallback):
    """
    Evaluates real-typo top-1 / top-5 against the model every `eval_every` steps.

    Args:
        eval_jsonl: path to a JSON list of {"typed": str, "committed": str} pairs.
        sp_model_path: SentencePiece tokenizer path.
        eval_every: steps between evals (default 500).
        max_pairs: cap on pairs evaluated per check (default 50).
        beams: beam count for top-5 (default 5).
        max_new: max generated tokens per query (default 20).
        csv_path: optional path to append CSV rows: step,top1,top5,n
    """
    def __init__(self, eval_jsonl: str, sp_model_path: str,
                 eval_every: int = 500, max_pairs: int = 50,
                 beams: int = 5, max_new: int = 20,
                 csv_path: str | None = None):
        self.eval_pairs = json.loads(Path(eval_jsonl).read_text())[:max_pairs]
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.xbu_id = self.sp.piece_to_id("<XBU>")
        self.xbc_id = self.sp.piece_to_id("<XBC>")
        self.xec_id = self.sp.piece_to_id("<XEC>")
        self.eval_every = eval_every
        self.beams = beams
        self.max_new = max_new
        self.csv_path = Path(csv_path) if csv_path else None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                self.csv_path.write_text("step,top1,top5,n\n")
        print(f"[real_eval] loaded {len(self.eval_pairs)} pairs from {eval_jsonl}, "
              f"will eval every {eval_every} steps", flush=True)

    def _encode_prompt(self, typo: str) -> list[int]:
        ids = [self.xbu_id]
        for tok in _to_keypress_chars(typo):
            ids.append(self.sp.piece_to_id(tok))
        ids.append(self.xbc_id)
        return ids

    def _decode_until_xec(self, gen_ids: list[int]) -> str:
        cut = gen_ids.index(self.xec_id) if self.xec_id in gen_ids else len(gen_ids)
        return self.sp.decode(gen_ids[:cut]).strip()

    def _run_eval(self, model) -> tuple[int, int, int]:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        top1 = top5 = n = 0
        try:
            with torch.no_grad():
                for pair in self.eval_pairs:
                    typo = pair.get("typed") or pair.get("typo")
                    correct = pair.get("committed") or pair.get("correct")
                    if not typo or not correct:
                        continue
                    n += 1
                    prompt_ids = self._encode_prompt(typo)
                    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    g_out = model.generate(
                        prompt, max_new_tokens=self.max_new, do_sample=False,
                        num_beams=1, pad_token_id=self.sp.pad_id(),
                        eos_token_id=self.xec_id,
                    )
                    t1 = self._decode_until_xec(g_out[0, len(prompt_ids):].tolist())
                    b_out = model.generate(
                        prompt, max_new_tokens=self.max_new, do_sample=False,
                        num_beams=self.beams, num_return_sequences=self.beams,
                        pad_token_id=self.sp.pad_id(), eos_token_id=self.xec_id,
                    )
                    beams = [self._decode_until_xec(b_out[k, len(prompt_ids):].tolist())
                             for k in range(min(self.beams, b_out.size(0)))]
                    if t1.strip() == correct.strip():
                        top1 += 1
                    if any(b.strip() == correct.strip() for b in beams[:5]):
                        top5 += 1
        finally:
            if was_training:
                model.train()
        return top1, top5, n

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return
        if model is None:
            return
        try:
            top1, top5, n = self._run_eval(model)
        except Exception as e:
            print(f"[real_eval] step={state.global_step} ERROR: {type(e).__name__}: {e}",
                  flush=True)
            return
        print(f"[real_eval] step={state.global_step} top1={top1}/{n} top5={top5}/{n}",
              flush=True)
        if self.csv_path:
            with open(self.csv_path, "a") as f:
                f.write(f"{state.global_step},{top1},{top5},{n}\n")
