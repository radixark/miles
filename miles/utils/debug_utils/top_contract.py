"""A module of TOP's backward contract as its family reference declares it (top_reference, judged by top_vjp)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import torch


class ContractModule(NamedTuple):
    """One TOP module the contract check replays at TOP's captured inputs: its declared surrogate in the reference."""

    # The reference layer's parameters this module owns: name prefixes ("attn." for a submodule) or full names.
    params: tuple[str, ...]
    # (reference model, reference layer, x [1, n, W], positions [1, n], causal [1, 1, n, n]) -> the surrogate's
    # outputs, one [1, n, ...] tensor per captured output, in ``outputs`` order.
    run: Callable[..., tuple[torch.Tensor, ...]]
    # TOP's captured outputs (top_capture kinds after the module name).
    outputs: tuple[str, ...] = ("output",)
    # Whether TOP's captured input gradient is this module's alone. An mHC module's input also feeds the residual
    # path after it, so its captured gradient carries more than the module's VJP.
    judges_input_grad: bool = True
    # The run also takes ``carried=``: the state the layer carries outside its streams (top_reference.Carried).
    takes_carried: bool = False

    def owns(self, name: str) -> bool:
        return name.startswith(self.params)


def _submodule_run(name: str, run: Callable[..., torch.Tensor], model, layer, x, positions, causal):
    return (run(model, layer.get_submodule(name), x, positions, causal),)


def submodule(name: str, run: Callable[..., torch.Tensor]) -> ContractModule:
    """A module whose surrogate is the reference submodule ``name`` with one output: run(model, submodule, x,
    positions, causal) -> [1, n, H]."""
    return ContractModule(params=(f"{name}.",), run=functools.partial(_submodule_run, name, run))
