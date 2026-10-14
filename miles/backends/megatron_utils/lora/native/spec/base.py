"""Architecture contracts shared by native-LoRA specs, modules, and exporters."""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig


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

    ``attr`` names the parameter pair stored on the adapter
    (``<attr>_A``/``<attr>_B``). The exporters derive each rank's shard width from
    those parameter shapes, so this descriptor stays static and pickle-safe.
    """

    hf: str
    attr: str
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
    naming: Any
    full_sequence: bool = False

    @property
    def rank(self) -> int:
        return self.lora.rank

    @property
    def scale(self) -> float:
        return self.lora.scale

    @property
    def dropout(self) -> float:
        return self.lora.dropout

    @property
    def a_init(self) -> str:
        return self.lora.a_init_method

    @property
    def hidden(self) -> int:
        return self.transformer_config.hidden_size

    @property
    def sequence_parallel(self) -> bool:
        """Whether the adapted hosts see sequence-parallel shards; a full-sequence block gathers them first."""
        return not self.full_sequence and bool(self.transformer_config.sequence_parallel)

    @property
    def output_gate(self) -> bool:
        return bool(getattr(self.transformer_config, "attention_output_gate", False))

    def selects(self, hf_module: str) -> bool:
        return self.lora.selects(hf_module)


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
    model_family: str
    attention: AttentionLoRASpec
    mlp: MLPLoRASpec
    experts: ExpertsLoRASpec | None = None
    lm_head: Any = None
    allows_mixer_only_adapter_chunks: bool = False
    fixed_targets: FixedTargets | None = None
    # None names adapters after the raw-mode converter's HF weight names
    declared_naming: Any = None

    def attaches(self, hf_module: str) -> bool:
        """Whether this spec implements an adapter for the HF module (``*`` stands for any layer)."""
        block, _, leaf = hf_module.rpartition(".")
        if leaf in self.attention.supported_targets:
            return re.search(f"{_HF_LAYER}{re.escape(self.attention.hf_block)}$", block) is not None
        if leaf in self.mlp.supported_targets:
            return re.search(rf"{_HF_LAYER}mlp(?:\.shared_experts?)?$", block) is not None
        return False

    def serving_fused_families(self) -> list[frozenset[str]]:
        """Projection families SGLang stores in one fused buffer."""
        return [*self.attention.serving_fused_families(), *self.mlp.serving_fused_families()]

    def validate(self, context: AttachContext) -> None:
        assert context.transformer_config.gated_linear_unit, "native LoRA's MLP specs assume a gated (SwiGLU) MLP"
        self.attention.validate(context.transformer_config, tp_size=context.tp_size)
