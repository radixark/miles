"""Architecture contracts shared by native-LoRA specs, modules, and exporters."""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
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
    def targets(self) -> frozenset[str]:
        return self.lora.target_modules

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

    def wants(self, *names: str) -> bool:
        return bool(self.targets.intersection(names))


class AttentionLoRASpec(Protocol):
    """Architecture-specific attention attachment contract."""

    name: str
    supported_targets: frozenset[str]

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]: ...

    def validate(self, config, *, tp_size: int) -> None: ...

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int: ...


class MLPLoRASpec(Protocol):
    """Architecture-specific dense/shared-MLP attachment contract."""

    name: str
    supported_targets: frozenset[str]

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int: ...


class MoELoRASpec(Protocol):
    """Routed-expert validation/attachment boundary.

    ``validate_layer`` returns the MLP targets the layer cannot attach (for the
    orchestrator to report once per run) and raises when the miss is an error.
    """

    supported_targets: frozenset[str]

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> frozenset[str]: ...


@dataclass(frozen=True)
class LoRAArchSpec:
    """Complete native-LoRA contract selected for one HF model architecture."""

    name: str
    model_family: str
    attention: AttentionLoRASpec
    mlp: MLPLoRASpec
    moe: MoELoRASpec
    lm_head: Any = None
    allows_mixer_only_adapter_chunks: bool = False
    sglang_lora_target_modules: tuple[str, ...] | None = None

    @property
    def supported_targets(self) -> frozenset[str]:
        return self.attention.supported_targets | self.mlp.supported_targets | self.moe.supported_targets

    def normalize_config(self, config: LoRAConfig) -> LoRAConfig:
        """Apply architecture-specific compatibility normalization to targets."""
        targets = self.attention.normalize_targets(
            config.target_modules,
            expanded_from_all_linear=config.expanded_from_all_linear,
        )
        assert targets, (
            f"native LoRA architecture spec {self.name!r} has no effective target modules after "
            "architecture normalization; select a projection implemented by this model family"
        )
        return config if targets == config.target_modules else replace(config, target_modules=targets)

    def validate_targets(self, targets: frozenset[str]) -> None:
        """Fail closed on projection names before model/runtime metadata exists."""
        unsupported = sorted(targets - self.supported_targets)
        assert not unsupported, (
            f"native LoRA architecture spec {self.name!r} does not implement targets {unsupported}. "
            f"Supported targets are {sorted(self.supported_targets)}; use --megatron-to-hf-mode bridge "
            "or point --lora-provider-path at a model-specific provider."
        )

    def validate(self, context: AttachContext) -> None:
        self.validate_targets(context.targets)
        self.attention.validate(context.transformer_config, tp_size=context.tp_size)
