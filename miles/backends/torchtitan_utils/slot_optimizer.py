"""Per-tenant slot optimizers for torchtitan multi-LoRA.

Each live slot owns a torchtitan ``AdamW`` over exactly its adapter parameters (DTensors under FSDP), built at load
and dropped with the slot, so a fresh tenant never inherits optimizer state. The update is the one torchtitan's
single-LoRA ``Optim`` makes: ``dist_utils.clip_grad_norm_`` then a fused AdamW step.
"""

import math

import torch
from torchtitan.components.optim import AdamW
from torchtitan.distributed import utils as dist_utils


class TorchtitanSlotOptimizer:
    """One tenant's optimizer over one adapter slot; the parameters are their own fp32 masters."""

    def __init__(self, slot: int, slot_params: list[torch.nn.Parameter]) -> None:
        assert slot_params, f"adapter slot {slot} has no parameters; is this a multi-LoRA model?"
        self.slot = slot
        self.params = slot_params
        # the hyperparameters arrive with every step (apply_adam_params)
        self.optimizer = AdamW(AdamW.Config(pattern=r".*"), params=slot_params)

    def apply_adam_params(self, adam_params: dict) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = adam_params["learning_rate"]
            group["betas"] = (adam_params["beta1"], adam_params["beta2"])
            group["eps"] = adam_params["eps"]
            group["weight_decay"] = adam_params["weight_decay"]

    def clip_and_step(self, clip_grad: float) -> dict:
        """-> {"grad_norm": x} stepped, {"skipped_nonfinite": 1.0} dropped.

        Collective: the slot's grad norm reduces over the parameters' meshes. ``clip_grad`` 0 clips nothing.
        """
        grad_norm = dist_utils.clip_grad_norm_(
            self.params, clip_grad if clip_grad > 0.0 else math.inf, foreach=True, pp_mesh=None
        ).item()
        # the reduced norm is the same on every rank, so every rank skips alike
        if not math.isfinite(grad_norm):
            return {"skipped_nonfinite": 1.0}
        self.optimizer.step()
        return {"grad_norm": grad_norm}

    def reset_adam_state(self) -> None:
        """Drop the Adam moments and step, keeping the weights: the state of a slot that never stepped."""
        self.optimizer.state.clear()

    def zero_grads(self) -> None:
        for param in self.params:
            param.grad = None


def step_slot_optimizers(
    slot_optimizers: dict[int, TorchtitanSlotOptimizer], adam_params_by_slot: dict[int, dict]
) -> dict[int, dict]:
    """Step slots in the same order on every rank, then drop their gradients."""
    slots = sorted(adam_params_by_slot)
    outcomes = {}
    for slot in slots:
        slot_optimizers[slot].apply_adam_params(adam_params_by_slot[slot])
        outcomes[slot] = slot_optimizers[slot].clip_and_step(adam_params_by_slot[slot]["grad_clip_norm"])
    for slot in slots:
        slot_optimizers[slot].zero_grads()
    return outcomes
