"""The numerical program, as data. One declaration per model family; both engines resolve from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


# delegate      trainer runs the rollout's op object
# match         selected fused implementations agree; no delegated forward is needed
# batch_invariant  both engines use a batch-invariant reduction proven to agree
# match_native  they agree only once megatron's conflicting fused path is disabled; the rollout
#               may still run its own fused kernel
Verb = Literal["delegate", "match", "batch_invariant", "match_native"]


@dataclass(frozen=True)
class NormParams:
    """Parameters of sglang's RMSNorm expression."""

    weight_dtype: torch.dtype | None = None
    override_orig_dtype: torch.dtype | None = None
    fp32_residual: bool = False


@dataclass(frozen=True)
class Program:
    name: str
    norm: dict[str, NormParams] = field(default_factory=dict)
    ops: dict[str, Verb] = field(default_factory=dict)


# bf16 stream, fp32 internal -- what both engines run by default
QWEN3_DENSE_V1 = Program(
    name="qwen3-dense/v1",
    norm={r: NormParams() for r in (
        "input_layernorm", "pre_mlp_layernorm", "final_layernorm",
        "q_layernorm", "k_layernorm",
    )},
    ops={
        "norm": "delegate",
        "row_linear": "delegate",
        "attention": "delegate",
        "matmul": "batch_invariant",  # validated for the qualified shapes and runtime
        "activation": "match_native",
        "rope": "match",
    },
)


_BY_NAME = {QWEN3_DENSE_V1.name: QWEN3_DENSE_V1}

def get_program(name: str) -> Program:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"[top] unknown program {name!r}; known: {sorted(_BY_NAME)}") from None
