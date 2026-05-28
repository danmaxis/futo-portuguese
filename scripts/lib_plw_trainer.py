"""
Trainer subclass + helpers for Prompt Loss Weight (PLW) training.

PLW is the per-token weight applied to clean-text (non-completion) tokens when
computing the loss. Setting PLW=0 fully masks them (equivalent to labels=-100).
Setting PLW=1 trains on every token with equal weight. The literature
(arxiv 2401.13586 "Does Prompt Loss Matter?") shows that for SHORT completions
(~5 tokens, like our XBU correction spans), the optimal PLW is in [0, 0.1],
with PLW=1 (full-sequence loss) being significantly worse — that's the
diagnosed root cause of our stage_b/c mode collapse.

Training datasets in 04a/04b emit a `loss_weights` tensor alongside input_ids
and labels. PLWTrainer reads it and computes the weighted loss.

SAMPLWTrainer adds Sharpness-Aware Minimization (Foret et al. 2020) on top of
PLW for catastrophic-forgetting mitigation — relevant for our sequential
4a → 4b → 4c regime where each stage tends to overwrite the previous.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import Trainer

from lib_sam import SAMWrapper


class PLWTrainer(Trainer):
    """
    Custom Trainer that computes per-token weighted cross-entropy loss using a
    `loss_weights` tensor provided by the dataset.

    Dataset weight conventions:
      - 1.0 for tokens we want to fully learn (the correction / completion)
      - plw (e.g. 0.05) for clean-text / prompt tokens
      - 0.0 for pad or ignore tokens (labels=-100 also still works)
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # NOTE: do NOT pop from inputs — SAMPLWTrainer calls compute_loss twice
        # with the same inputs dict per micro-batch.
        labels = inputs["labels"]
        loss_weights = inputs.get("loss_weights")
        model_inputs = {k: v for k, v in inputs.items()
                        if k not in ("labels", "loss_weights")}
        outputs = model(**model_inputs)
        logits = outputs.logits  # (B, T, V)

        # Standard causal-LM shift: predict labels[i+1] from logits[i]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())  # (B, T-1)

        valid = (shift_labels != -100).float()

        if loss_weights is not None:
            # Align weights with shift_labels: the weight at position t+1 applies
            # to the loss computed for predicting labels[t+1].
            shift_weights = loss_weights[..., 1:].contiguous().to(ce.dtype)
            weighted = ce * shift_weights * valid
            denom = (shift_weights * valid).sum().clamp_min(1e-8)
            loss = weighted.sum() / denom
        else:
            loss = (ce * valid).sum() / valid.sum().clamp_min(1e-8)

        return (loss, outputs) if return_outputs else loss


class SAMPLWTrainer(PLWTrainer):
    """
    PLWTrainer + Sharpness-Aware Minimization.

    Each training step does TWO forward+backward passes:
      1. At current weights w, compute grad g
      2. Perturb to w + rho*g/||g||, compute grad g_sam
      3. Restore w, let HF's optimizer.step() apply g_sam

    Doubles per-step cost but produces flatter minima → less catastrophic
    forgetting in the 4a → 4b transition.

    Grad-accum interaction: SAM perturbs/restores per micro-batch. With
    grad_accum > 1, this wastes some compute but keeps integration simple.
    Recommended: use grad_accum=1 with SAM and a larger micro-batch.
    """
    def __init__(self, *args, sam_rho: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._sam = SAMWrapper(rho=sam_rho)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Pass 1: gradient at current weights
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        loss_for_return = loss.detach()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        # SAM perturb uphill
        self._sam.first_step(model)

        # Pass 2: clear grad, recompute at perturbed weights
        self.optimizer.zero_grad()
        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss2 = loss2 / self.args.gradient_accumulation_steps
        loss2.backward()

        # Restore original weights; HF's optimizer.step() then applies g_sam
        self._sam.second_step(model)

        return loss_for_return


def build_loss_weights_for_xbu(token_ids, xbu_id: int, xec_id: int,
                                plw_clean: float = 0.05,
                                in_span_weight: float = 1.0):
    """
    Inline-corrupted (04b) style: mark tokens inside XBU..XEC spans (the typo
    keypress + correction region) with full weight; everything else with plw_clean.
    """
    weights = []
    in_span = False
    for t in token_ids:
        t = int(t)
        if t == xbu_id:
            in_span = True
            weights.append(in_span_weight)
        elif t == xec_id:
            weights.append(in_span_weight)
            in_span = False
        elif in_span:
            weights.append(in_span_weight)
        else:
            weights.append(plw_clean)
    return weights


def build_loss_weights_for_correction_only(token_ids, xbc_id: int, xec_id: int,
                                             plw_clean: float = 0.05,
                                             in_span_weight: float = 1.0):
    """
    Isolated-triple (04a) style: mark tokens from <XBC> through <XEC> as the
    correction span (full weight). Everything else (BOS, XBU, CHAR_*, padding)
    gets plw_clean. With plw_clean=0 this reduces to the old labels=-100 mask.
    """
    weights = [plw_clean] * len(token_ids)
    found_xbc = False
    for i, t in enumerate(token_ids):
        t = int(t)
        if t == xbc_id:
            found_xbc = True
        if found_xbc:
            weights[i] = in_span_weight
        if t == xec_id and found_xbc:
            # Continue marking any remaining padding/eos with plw_clean
            for j in range(i + 1, len(token_ids)):
                weights[j] = plw_clean
            break
    return weights
