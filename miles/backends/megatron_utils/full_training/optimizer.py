"""Accumulate Tinker commands in Megatron's optimizer-owned gradient shards."""

import math

import torch


class GradientAccumulator:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.gradients: dict[torch.nn.Parameter, torch.Tensor] = {}
        self.num_batches = 0

    @torch.no_grad()
    def add(self) -> None:
        # After reduce-scatter, only the optimizer's shard is valid. Keeping the
        # entire DDP buffer across commands would reduce stale shards again.
        self.optimizer.prepare_grads()
        for param in self.optimizer.get_parameters():
            if param.grad is not None:
                if param in self.gradients:
                    self.gradients[param].add_(param.grad)
                else:
                    self.gradients[param] = param.grad.detach().clone()
        self.num_batches += 1

    @torch.no_grad()
    def step(self, adam: dict) -> dict:
        if not self.num_batches:
            return {"error": "optim_step requires accumulated gradients"}
        for group in self.optimizer.param_groups:
            group.update(
                lr=adam["learning_rate"],
                betas=(adam["beta1"], adam["beta2"]),
                eps=adam["eps"],
                weight_decay=adam["weight_decay"],
            )
        for param in self.optimizer.get_parameters():
            param.grad = self.gradients.get(param)
        grad_norm = float(self.optimizer.get_grad_norm())
        if not math.isfinite(grad_norm):
            self.clear()
            return {"skipped_nonfinite": 1}
        if adam["grad_clip_norm"] > 0:
            coefficient = min(1.0, adam["grad_clip_norm"] / (grad_norm + 1e-6))
            for gradient in self.gradients.values():
                gradient.mul_(coefficient)
        # Gradients have already been prepared by add(); step() would overwrite
        # them from the last command's DDP buffers and discard earlier commands.
        successful = self.optimizer.step_with_ready_grads()
        self.clear()
        return {"grad_norm": grad_norm} if successful else {"skipped_nonfinite": 1}

    def clear(self) -> None:
        self.optimizer.zero_grad()
        self.gradients.clear()
        self.num_batches = 0
