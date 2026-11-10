from tests.ci.ci_register import register_cuda_ci

# CPU-only checks, on this stage because only the GPU image ships torchtitan
register_cuda_ci(
    est_time=10, suite="stage-b-2-gpu-h200", labels=["torchtitan", "lora"], hardware=["hopper"], num_gpus=2
)

import math
import sys

import pytest
import torch

# the package applies the torch 2.14 backports torchtitan needs at import, so it must load first
import miles.backends.torchtitan_utils  # noqa: F401

# isort: split
from torchtitan.components.optim import AdamW
from torchtitan.distributed import utils as dist_utils

from miles.backends.torchtitan_utils.slot_optimizer import TorchtitanSlotOptimizer, step_slot_optimizers

# none of them AdamW.Config's defaults, so a hyperparameter left unapplied shows
_ADAM = {"learning_rate": 1e-2, "beta1": 0.8, "beta2": 0.98, "eps": 1e-6, "weight_decay": 0.03, "grad_clip_norm": 0.5}


def _params(seed: int) -> list[torch.nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(shape, generator=g)) for shape in ((4, 16), (32, 4), (2, 8, 4))]


def _grads(params, seed: int, scale: float = 1.0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g) * scale


def test_a_slot_steps_like_torchtitans_clip_and_adamw():
    params, reference = _params(0), _params(0)
    optimizers = {3: TorchtitanSlotOptimizer(3, params)}
    adamw = AdamW(
        AdamW.Config(pattern=r".*", lr=_ADAM["learning_rate"], betas=(0.8, 0.98), eps=1e-6, weight_decay=0.03),
        params=reference,
    )
    for step in range(3):
        _grads(params, seed=step, scale=10.0)
        _grads(reference, seed=step, scale=10.0)
        outcome = step_slot_optimizers(optimizers, {3: _ADAM})[3]
        norm = dist_utils.clip_grad_norm_(reference, _ADAM["grad_clip_norm"], foreach=True)
        adamw.step()
        assert outcome == {"grad_norm": norm.item()}
        assert all(torch.equal(p, r) for p, r in zip(params, reference, strict=True))
        assert all(p.grad is None for p in params)


def test_a_nonfinite_gradient_skips_the_step_and_drops_the_gradients():
    params = _params(1)
    before = [p.detach().clone() for p in params]
    optimizer = TorchtitanSlotOptimizer(0, params)
    _grads(params, seed=1)
    params[1].grad[0, 0] = math.nan
    assert step_slot_optimizers({0: optimizer}, {0: _ADAM}) == {0: {"skipped_nonfinite": 1.0}}
    assert all(torch.equal(p, b) for p, b in zip(params, before, strict=True))
    assert all(p.grad is None for p in params) and not optimizer.optimizer.state


def test_clip_zero_leaves_the_gradients_unclipped():
    params = _params(2)
    optimizer = TorchtitanSlotOptimizer(0, params)
    _grads(params, seed=2, scale=100.0)
    grads = [p.grad.clone() for p in params]
    outcome = optimizer.clip_and_step(0.0)
    assert outcome["grad_norm"] == pytest.approx(
        torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads])).item()
    )
    assert all(torch.equal(p.grad, g) for p, g in zip(params, grads, strict=True))


def test_only_the_named_slots_step():
    stepped, idle = _params(3), _params(4)
    idle_before = [p.detach().clone() for p in idle]
    optimizers = {0: TorchtitanSlotOptimizer(0, stepped), 1: TorchtitanSlotOptimizer(1, idle)}
    _grads(stepped, seed=5)
    _grads(idle, seed=6)
    assert set(step_slot_optimizers(optimizers, {0: _ADAM})) == {0}
    assert all(torch.equal(p, b) for p, b in zip(idle, idle_before, strict=True))
    # an unstepped slot keeps its accumulated gradients for its own step
    assert all(p.grad is not None for p in idle)


def test_reset_drops_the_moments_and_keeps_the_weights():
    params = _params(7)
    optimizer = TorchtitanSlotOptimizer(0, params)
    _grads(params, seed=7)
    step_slot_optimizers({0: optimizer}, {0: _ADAM})
    weights = [p.detach().clone() for p in params]
    assert optimizer.optimizer.state
    optimizer.reset_adam_state()
    assert not optimizer.optimizer.state
    assert all(torch.equal(p, w) for p, w in zip(params, weights, strict=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
