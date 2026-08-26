"""Fail-closed optimizer-state reset helpers for Megatron training."""

from collections.abc import MutableMapping, Sequence
from typing import Any

import torch
from megatron.core.optimizer import Adam as MegatronAdam
from megatron.core.optimizer import CPUAdam
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon


def _zero_step(container: MutableMapping[str, Any]) -> None:
    if "step" not in container:
        return

    step = container["step"]
    if isinstance(step, torch.Tensor):
        step.zero_()
    elif isinstance(step, (int, float)):
        container["step"] = 0
    else:
        raise TypeError(f"Unsupported optimizer step type: {type(step).__name__}")


def _zero_tensor(state: MutableMapping[str, Any], key: str) -> None:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor optimizer state {key!r}, got {type(value).__name__}")
    value.zero_()


def _reset_adam_state(optimizer: torch.optim.Optimizer) -> None:
    # Some Adam implementations store the step inside optimizer.param_groups.
    for group in optimizer.param_groups:
        _zero_step(group)

    for state in optimizer.state.values():
        unexpected_keys = set(state) - {"step", "exp_avg", "exp_avg_sq", "master_param"}
        if unexpected_keys:
            raise RuntimeError(f"Unsupported Adam optimizer state keys: {sorted(unexpected_keys)}")

        has_exp_avg = "exp_avg" in state
        has_exp_avg_sq = "exp_avg_sq" in state

        # Both moments may be absent before a parameter's first optimizer step or for frozen parameters.
        if has_exp_avg != has_exp_avg_sq:
            raise RuntimeError(f"Incomplete Adam optimizer state: {sorted(state)}")

        # This is a no-op for implementations that store the step only in param_groups.
        _zero_step(state)

        if has_exp_avg:
            _zero_tensor(state, "exp_avg")
            _zero_tensor(state, "exp_avg_sq")


def _reset_muon_state(optimizer: TensorParallelMuon) -> None:
    for state in optimizer.state.values():
        unexpected_keys = set(state) - {"momentum_buffer"}
        if unexpected_keys:
            raise RuntimeError(f"Unsupported Muon optimizer state keys: {sorted(unexpected_keys)}")
        if "momentum_buffer" in state:
            _zero_tensor(state, "momentum_buffer")


def _reset_supported_adam_state(optimizer: torch.optim.Optimizer) -> None:
    if isinstance(optimizer, HybridDeviceOptimizer):
        if optimizer.gpu_optimizer is not None:
            raise TypeError("HybridDeviceOptimizer with a GPU child optimizer is not supported")
        if not optimizer.cpu_optimizers:
            raise TypeError("HybridDeviceOptimizer has no CPU child optimizers")
        for child_optimizer in optimizer.cpu_optimizers:
            if not isinstance(child_optimizer, CPUAdam):
                raise TypeError(f"HybridDeviceOptimizer wraps unsupported CPU optimizer {type(child_optimizer).__name__}")
    elif not isinstance(optimizer, (MegatronAdam, CPUAdam)):
        raise TypeError(f"Unsupported Adam optimizer implementation: {type(optimizer).__name__}. When adding support for a new optimizer, update _reset_adam_state() for its state schema")

    _reset_adam_state(optimizer)


def reset_optimizer_states(optimizer_name: str, optimizers: Sequence[torch.optim.Optimizer]) -> None:
    """Reset all history for a supported optimizer, failing on unknown implementations or state."""
    if optimizer_name == "adam":
        for optimizer in optimizers:
            _reset_supported_adam_state(optimizer)
        return

    if optimizer_name == "dist_muon":
        for optimizer in optimizers:
            if isinstance(optimizer, TensorParallelMuon):
                _reset_muon_state(optimizer)
            else:
                _reset_supported_adam_state(optimizer)
        return

    raise NotImplementedError(f"--reset-optimizer-states does not support optimizer {optimizer_name!r}")
