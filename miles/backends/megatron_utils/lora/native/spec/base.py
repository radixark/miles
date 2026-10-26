"""Architecture contracts shared by native-LoRA specs, modules, and exporters."""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.naming import HFNaming


class ShardLayout(str, enum.Enum):
    """How one logical projection is sharded across the tensor-parallel group."""

    COLUMN = "column"
    ROW = "row"
    REPLICATED = "replicated"


class AttentionFamily(str, enum.Enum):
    """Structural attention family a registry entry belongs to."""

    GQA = "gqa"
    MLA = "mla"


@dataclass(frozen=True)
class ProjectionSpec:
    """External name and shard layout for one logical HF LoRA projection.

    ``hf_leaf`` is the projection's HF module name inside its block (``q_proj``).
    ``param_prefix`` names the parameter pair stored on the adapter
    (``<param_prefix>_A``/``<param_prefix>_B``). The exporters derive each rank's shard
    width from those parameter shapes, so this descriptor stays static and pickle-safe.
    """

    hf_leaf: str
    param_prefix: str
    layout: ShardLayout


@dataclass(frozen=True)
class AttachContext:
    """Resolved runtime information passed to architecture attachment specs.

    Deliberately keeps run-level ``LoRAConfig`` separate from model and parallel metadata.
    """

    lora: LoRAConfig
    transformer_config: Any
    tp_size: int
    tp_rank: int
    naming: HFNaming
    full_sequence: bool = False

    @property
    def sequence_parallel(self) -> bool:
        """Whether the adapted hosts see sequence-parallel shards; a full-sequence block gathers them first."""
        return not self.full_sequence and bool(self.transformer_config.sequence_parallel)

    @property
    def output_gate(self) -> bool:
        return bool(
            getattr(self.transformer_config, "attention_output_gate", False)
        )  # config-access-exempt: only attention architectures with output gates declare this config field


class AttentionLoRASpec(Protocol):
    """Architecture-specific attention attachment contract."""

    name: str
    hf_block: str
    supported_targets: frozenset[str]

    def validate(self, config, *, tp_size: int) -> None: ...

    def applies_to(self, attention: nn.Module) -> bool: ...

    def attach(self, attention: nn.Module, layer: nn.Module, context: AttachContext) -> int: ...


class MLPLoRASpec(Protocol):
    """Architecture-specific dense/shared-MLP attachment contract."""

    name: str
    supported_targets: frozenset[str]

    def attach(self, mlp: nn.Module, layer: nn.Module, context: AttachContext) -> int: ...


class ExpertsLoRASpec(Protocol):
    """Routed-expert attachment for architectures whose experts carry adapters."""

    def attach(self, mlp: nn.Module, layer: nn.Module, context: AttachContext) -> int: ...


_HF_LAYER = r"(?:^|\.)layers\.(?:\d+|\*)\."


@dataclass(frozen=True)
class FixedTargets:
    """A verified adapter layout: requests must be the model's HF target layout, minus ``optional`` suffixes.

    ``serving`` turns the request into SGLang's target list. ``select_all`` attaches every
    implemented projection regardless of names, for specs whose export names are not HF targets.
    """

    serving: Callable[[list[str]], list[str] | str]
    optional: frozenset[str] = frozenset()
    select_all: bool = False


@dataclass(frozen=True)
class LoRAArchSpec:
    """Complete native-LoRA contract selected for one HF model architecture.

    Without ``fixed_targets``, any selection of attachable projections is accepted.
    """

    name: str
    attention: AttentionLoRASpec
    mlp: MLPLoRASpec
    experts: ExpertsLoRASpec | None = None
    lm_head: Any = None
    has_linear_attention_layers: bool = False
    fixed_targets: FixedTargets | None = None
    # None names adapters after the raw-mode converter's HF weight names
    declared_naming: Any = None

    def attaches(self, hf_module: str) -> bool:
        """Whether this spec implements an adapter for the HF module (``*`` stands for any layer)."""
        blocks = (
            (re.escape(self.attention.hf_block), self.attention.supported_targets),
            (r"mlp(?:\.shared_experts?)?", self.mlp.supported_targets),
        )
        # a target may nest below its block (GLM-5.3-Flash's forget_gate.f_a_proj)
        return any(
            re.search(rf"{_HF_LAYER}{block}\.{re.escape(target)}$", hf_module)
            for block, targets in blocks
            for target in targets
        )

    def validate(self, context: AttachContext) -> None:
        assert context.transformer_config.gated_linear_unit, "native LoRA's MLP specs assume a gated (SwiGLU) MLP"
        self.attention.validate(context.transformer_config, tp_size=context.tp_size)
