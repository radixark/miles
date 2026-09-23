"""Reset optimizer history for --reset-optimizer-states, failing closed on state it does not know.

The Muon classes are imported where they are matched: Megatron builds without
``emerging_optimizers`` (the NPU pin) still run Adam.
"""

from collections.abc import Iterator

import torch
from megatron.core.optimizer import Adam, CPUAdam


class UnsupportedOptimizerState(RuntimeError):
    """An optimizer or state key this module has no schema for; teach _history_keys rather than guessing."""


def reset_optimizer_states(optimizer) -> None:
    for leaf in _leaves(optimizer):
        _reset(leaf, _history_keys(leaf))


def _leaves(optimizer) -> Iterator[torch.optim.Optimizer]:
    """Descend Megatron's wrappers to the torch optimizers that own the state."""
    if optimizer is None:
        return
    for children in ("chained_optimizers", "sub_optimizers"):
        if hasattr(optimizer, children):
            for child in getattr(optimizer, children):
                yield from _leaves(child)
            return
    if hasattr(optimizer, "optimizer"):
        yield from _leaves(optimizer.optimizer)
    else:
        yield optimizer


def _history_keys(optimizer) -> frozenset[str]:
    if isinstance(optimizer, (Adam, CPUAdam)):
        return frozenset({"exp_avg", "exp_avg_sq"})
    from megatron.core.optimizer.emerging_optimizers import TensorParallelAdaptiveMuon, TensorParallelMuon

    if isinstance(optimizer, TensorParallelAdaptiveMuon):
        return frozenset({"momentum_buffer", "moment2_buffer"})
    if isinstance(optimizer, TensorParallelMuon):
        return frozenset({"momentum_buffer"})
    raise UnsupportedOptimizerState(f"no reset schema for {type(optimizer).__name__}")


def _reset(optimizer, history: frozenset[str]) -> None:
    for group in optimizer.param_groups:
        _reset_step(group)
    for state in optimizer.state.values():
        unknown = set(state) - history - {"step", "master_param"}
        if unknown:
            raise UnsupportedOptimizerState(f"unknown state keys {sorted(unknown)} on {type(optimizer).__name__}")
        _reset_step(state)
        for key in history & set(state):
            state[key].zero_()


def _reset_step(container) -> None:
    step = container.get("step")
    if isinstance(step, torch.Tensor):
        step.zero_()
    elif isinstance(step, (int, float)):
        container["step"] = 0
    elif step is not None:
        raise UnsupportedOptimizerState(f"unsupported step type {type(step).__name__}")
