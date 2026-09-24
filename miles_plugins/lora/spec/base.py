"""Architecture contracts shared by native-LoRA specs, modules, and exporters."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn

from miles_plugins.lora.config import LoRAConfig


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
    layer_prefix: str
    shared_expert: str

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
    def eps(self) -> float:
        return self.transformer_config.layernorm_epsilon

    @property
    def hidden(self) -> int:
        return self.transformer_config.hidden_size

    @property
    def sequence_parallel(self) -> bool:
        return bool(self.transformer_config.sequence_parallel)

    @property
    def zero_centered_gamma(self) -> bool:
        return bool(getattr(self.transformer_config, "layernorm_zero_centered_gamma", False))

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

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int: ...


class MLPLoRASpec(Protocol):
    """Architecture-specific dense/shared-MLP attachment contract."""

    name: str
    supported_targets: frozenset[str]

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int: ...


class ExpertsLoRASpec(Protocol):
    """Routed-expert attachment for architectures whose experts carry adapters."""

    def attach(self, mlp: nn.Module, hf_layer_prefix: str, context: AttachContext) -> int: ...


_HF_LAYER = r"(?:^|\.)layers\.(?:\d+|\*)\."


@dataclass(frozen=True)
class LoRAArchSpec:
    """Complete native-LoRA contract selected for one HF model architecture.

    ``complete_layout`` specs attach every projection they implement, export names
    SGLang auto-detects, and accept only the model's complete HF target layout.
    """

    name: str
    model_family: str
    attention: AttentionLoRASpec
    mlp: MLPLoRASpec
    experts: ExpertsLoRASpec | None = None
    lm_head: Any = None
    allows_mixer_only_adapter_chunks: bool = False
    complete_layout: bool = False

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
        self.attention.validate(context.transformer_config, tp_size=context.tp_size)
