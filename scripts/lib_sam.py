"""
Sharpness-Aware Minimization (SAM) for HF Trainer.

Canonical formulation (Foret et al. 2020, arxiv 2010.01412):
  For each training step:
    1. Compute gradient g at current weights w
    2. Perturb: w' = w + rho * g / ||g||
    3. Compute gradient g_sam at w'
    4. Restore: w = w' - rho * g / ||g||  (i.e. back to original)
    5. Apply g_sam via the base optimizer's step()

This trains the model to converge to flat minima, which empirically reduces
catastrophic forgetting between training stages (arxiv 2024.findings-emnlp.249) —
relevant for our Phase 4b/4c where the format learned in 4a tends to get
overwritten.

Implementation note: we apply SAM per micro-batch. This wastes some compute
under gradient accumulation, but keeps the integration with HF Trainer simple.
"""
from __future__ import annotations
import torch


class SAMWrapper:
    """
    Stateful helper that performs the SAM first_step/second_step around the
    standard optimizer.step(). The base optimizer is NOT replaced — this wrapper
    just perturbs and restores weights around two backward passes.

    Usage in Trainer.training_step:
        # ... forward + backward (computes grad g at w) ...
        self.sam.first_step(model)         # w := w + rho * g / ||g||
        self.optimizer.zero_grad()
        # ... forward + backward again (computes g_sam at w') ...
        self.sam.second_step(model)         # restore w
        # HF Trainer's normal optimizer.step() then applies g_sam to original w
    """

    def __init__(self, rho: float = 0.05):
        self.rho = rho
        self._old_params: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self, model: torch.nn.Module):
        grad_norm = self._grad_norm(model)
        scale = self.rho / (grad_norm + 1e-12)
        for p in model.parameters():
            if p.grad is None:
                continue
            self._old_params[id(p)] = p.data.clone()
            p.data.add_(p.grad * scale)

    @torch.no_grad()
    def second_step(self, model: torch.nn.Module):
        for p in model.parameters():
            if id(p) in self._old_params:
                p.data.copy_(self._old_params[id(p)])
        self._old_params.clear()

    @torch.no_grad()
    def _grad_norm(self, model: torch.nn.Module) -> torch.Tensor:
        grads = [p.grad.detach().norm(p=2) for p in model.parameters() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(grads), p=2)
