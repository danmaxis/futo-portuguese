"""Quick debug: what does the reference English model actually output for an XBU prompt?"""
import sentencepiece as spm
from llama_cpp import Llama

sp = spm.SentencePieceProcessor()
sp.load("reference_model/extracted_spm.model")

llm = Llama(model_path="reference_model/ml4_1_f16_meta_fixed.gguf",
            n_ctx=512, n_threads=4, verbose=False, logits_all=True)

# Test: prompt the model with <XBU>teh<XBC>, look at top-10 logits
prompt = "<XBU>teh<XBC>"
ids = sp.encode(prompt, out_type=int)
print(f"prompt {prompt!r} -> ids {ids}")
print(f"  decoded back: {sp.decode(ids)!r}")
for i, t in enumerate(ids):
    print(f"  id {t} = piece {sp.id_to_piece(t)!r}")

llm.reset()
llm.eval(ids)
print(f"\nllm.n_tokens = {llm.n_tokens}")
print(f"llm.scores shape: {llm.scores.shape if hasattr(llm.scores, 'shape') else type(llm.scores)}")
last_logits = llm.scores[llm.n_tokens - 1]
print(f"last_logits shape: {last_logits.shape}, dtype: {last_logits.dtype}")
print(f"last_logits non-zero entries: {(last_logits != 0).sum()}")

import numpy as np
arr = np.asarray(last_logits)
top10 = np.argsort(-arr)[:10]
print(f"\nTop-10 next-token candidates after {prompt!r}:")
for tid in top10:
    print(f"  id={tid:5d}  logit={arr[tid]:7.3f}  piece={sp.id_to_piece(int(tid))!r}")

# Now try to greedy decode 8 tokens
print(f"\n--- Greedy decode 8 tokens ---")
out = []
for step in range(8):
    arr = np.asarray(llm.scores[llm.n_tokens - 1])
    next_id = int(arr.argmax())
    print(f"  step {step}: id={next_id} piece={sp.id_to_piece(next_id)!r}  logit={arr[next_id]:.3f}")
    out.append(next_id)
    llm.eval([next_id])
print(f"\nFull output decoded: {sp.decode(out)!r}")

# Also check what eos_id is and whether it's being hit early
print(f"\nsp.eos_id() = {sp.eos_id()}")
print(f"sp.bos_id() = {sp.bos_id()}")
print(f"<XEC> id = {sp.piece_to_id('<XEC>')}")
