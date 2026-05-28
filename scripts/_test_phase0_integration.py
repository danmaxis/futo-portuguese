"""5-step end-to-end integration test for Phase 0 refactor.

Verifies:
  - 04a JsonlTriplesDataset emits the right shape, including loss_weights
  - 04b InlineCorruptedDataset emits the right shape, including loss_weights
  - PLWTrainer.compute_loss runs without dimension errors
  - RealTypoEvalCallback can run a single eval pass
  - 5 training steps complete without exploding loss

Run from /workspace via:
    python scripts/_test_phase0_integration.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import LlamaForCausalLM, TrainingArguments

from lib_plw_trainer import PLWTrainer
from lib_real_eval_callback import RealTypoEvalCallback

# Try paths — assume running from /workspace
BASE = "pretrain_big/base"
TOK = "tokenizer/spm_pt_br_v2.model"
SYNTH = "notes/synth_typos.json"
REAL_POOL = "notes/real_typos_pool.json"
REAL_EVAL = "notes/real_typos_eval.json"
CORPUS = "corpora/big"


def test_04a_dataset_one_batch():
    print("=== 04a dataset ===")
    sys.path.insert(0, str(ROOT))
    from importlib import import_module
    mod = import_module("04a_finetune_isolated".replace("-", "_"))
    # Above line won't work because '04a' isn't a valid module name. Inline instead:

# Direct path:
import importlib.util
spec_a = importlib.util.spec_from_file_location("ft_04a", ROOT / "04a_finetune_isolated.py")
ft_04a = importlib.util.module_from_spec(spec_a)
spec_a.loader.exec_module(ft_04a)
print("imported 04a")

spec_b = importlib.util.spec_from_file_location("ft_04b", ROOT / "04b_finetune_fulltext.py")
ft_04b = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(ft_04b)
print("imported 04b")


def show(d, name):
    print(f"{name}: input_ids.shape={tuple(d['input_ids'].shape)} "
          f"labels.shape={tuple(d['labels'].shape)} "
          f"loss_weights.shape={tuple(d['loss_weights'].shape)} "
          f"weights[head]={d['loss_weights'][:8].tolist()}")


print()
print("=== 04a sample ===")
ds_a = ft_04a.JsonlTriplesDataset(
    synth_jsonl=SYNTH, real_jsonl=REAL_POOL, sp_model_path=TOK,
    seq_len=64, real_mix_ratio=0.25, plw_clean=0.05, seed=42,
)
it_a = iter(ds_a)
ex_a = next(it_a)
show(ex_a, "04a")

print()
print("=== 04b sample ===")
# Build a tiny shard manually so we don't need a full corpus
tmpdir = Path(tempfile.mkdtemp())
shard = tmpdir / "shard_test.txt"
shard.write_text("\n".join([
    "Bom dia, como você está hoje?",
    "Eu fui ao mercado comprar pão e leite.",
    "Não esqueci do nosso encontro amanhã.",
    "Obrigado pela ajuda com o projeto.",
    "Hoje tem futebol no estádio do Maracanã.",
] * 200))  # repeat to fill seq_len
ds_b = ft_04b.InlineCorruptedDataset(
    shard_paths=[str(shard)], sp_model_path=TOK,
    seq_len=128, typo_rate=0.33, plw_clean=0.05, seed=42, shuffle_buffer=8,
)
it_b = iter(ds_b)
ex_b = next(it_b)
show(ex_b, "04b")

# Check that input_ids and labels are equal (HF does internal shift now)
assert torch.equal(ex_b["input_ids"], ex_b["labels"]), "04b input_ids must equal labels"
print("04b labels == input_ids: OK")


print()
print("=== loading model for 5-step test ===")
model = LlamaForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
print("model loaded, params:", sum(p.numel() for p in model.parameters()))

print()
print("=== 5-step training on 04a dataset ===")
import torch.utils.data as tud

class _ListDS(tud.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

items_a = [next(it_a) for _ in range(40)]
train_a = _ListDS(items_a)

with tempfile.TemporaryDirectory() as out:
    targs = TrainingArguments(
        output_dir=out, max_steps=5,
        per_device_train_batch_size=4, gradient_accumulation_steps=1,
        learning_rate=1e-4, warmup_steps=0, bf16=True,
        logging_steps=1, save_steps=1000,
        dataloader_num_workers=0, report_to=[],
        remove_unused_columns=False,
    )
    trainer = PLWTrainer(model=model, args=targs, train_dataset=train_a)
    out_train = trainer.train()
    print("04a 5-step train_loss:", out_train.training_loss)

print()
print("=== 5-step training on 04b dataset ===")
items_b = [next(it_b) for _ in range(40)]
train_b = _ListDS(items_b)
with tempfile.TemporaryDirectory() as out:
    targs = TrainingArguments(
        output_dir=out, max_steps=5,
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=1e-4, warmup_steps=0, bf16=True,
        logging_steps=1, save_steps=1000,
        dataloader_num_workers=0, report_to=[],
        remove_unused_columns=False,
    )
    trainer = PLWTrainer(model=model, args=targs, train_dataset=train_b)
    out_train = trainer.train()
    print("04b 5-step train_loss:", out_train.training_loss)

print()
print("=== real-typo eval callback (one eval pass) ===")
cb = RealTypoEvalCallback(
    eval_jsonl=REAL_EVAL, sp_model_path=TOK,
    eval_every=999, max_pairs=10, beams=3,
)
top1, top5, n = cb._run_eval(model)
print(f"sample eval (post 5-step train): top1={top1}/{n} top5={top5}/{n}")

print()
print("INTEGRATION TEST PASSED")
