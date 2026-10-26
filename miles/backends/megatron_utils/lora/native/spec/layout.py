"""Declarative native-LoRA layouts: dimension resolvers, bindings, and the attach walk."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import (
    LoRALinear,
    NativeLoRAAdapter,
    attach_adapter_forward,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext, AttentionFamily, ProjectionSpec

DimFn = Callable[[nn.Module, AttachContext], int]


def hidden_size(_module: nn.Module, context: AttachContext) -> int:
    return context.transformer_config.hidden_size


def config_dim(field: str) -> DimFn:
    """A transformer-config field used verbatim (e.g. ``q_lora_rank``)."""

    def resolve(_module: nn.Module, context: AttachContext) -> int:
        value = getattr(
            context.transformer_config, field
        )  # config-access-exempt: the architecture layout names the transformer dimension field
        assert value, f"transformer config field {field!r} must be a positive dimension, got {value!r}"
        return int(value)

    return resolve


def host_in_local(host_attr: str) -> DimFn:
    """This rank's input width of the host linear ``host_attr``, read from its weight."""

    def resolve(module: nn.Module, _context: AttachContext) -> int:
        return getattr(module, host_attr).weight.shape[
            1
        ]  # config-access-exempt: the architecture layout names the projection host attribute

    return resolve


def host_out_local(host_attr: str) -> DimFn:
    """This rank's output width of the host linear ``host_attr``, read from its weight."""

    def resolve(module: nn.Module, _context: AttachContext) -> int:
        return getattr(module, host_attr).weight.shape[
            0
        ]  # config-access-exempt: the architecture layout names the projection host attribute

    return resolve


def gqa_o_in_local(module: nn.Module, _context: AttachContext) -> int:
    """Row-parallel o_proj input: this rank's query heads times head dim."""
    return module.num_attention_heads_per_partition * module.hidden_size_per_attention_head


def mla_q_up_out_local(module: nn.Module, _context: AttachContext) -> int:
    """Column-parallel q_b output: local heads times the full MLA query head dim."""
    return module.num_attention_heads_per_partition * module.q_head_dim


def mla_kv_down_out(_module: nn.Module, context: AttachContext) -> int:
    """Replicated kv_a output: compressed KV rank plus the shared RoPE key slice."""
    config = context.transformer_config
    return int(config.kv_lora_rank + config.qk_pos_emb_head_dim)


def mla_kv_up_out_local(module: nn.Module, context: AttachContext) -> int:
    """Column-parallel kv_b output: local heads times (nope-K plus V) head dims."""
    config = context.transformer_config
    return module.num_attention_heads_per_partition * (config.qk_head_dim + config.v_head_dim)


def mla_o_in_local(module: nn.Module, context: AttachContext) -> int:
    """Row-parallel MLA o_proj input: local heads times the value head dim."""
    return module.num_attention_heads_per_partition * context.transformer_config.v_head_dim


def inkling_o_in_local(module: nn.Module, _context: AttachContext) -> int:
    """Row-parallel Inkling wo_ud input: this rank's query heads times head dim."""
    return module.nh_l * module.hd


GuardFn = Callable[[nn.Module, AttachContext, ProjectionSpec, int], None]
FusedBuildFn = Callable[[nn.Module, str, AttachContext, tuple[ProjectionSpec, ...]], NativeLoRAAdapter]


@dataclass(frozen=True)
class ProjectionBinding:
    """One independently attachable projection and where it lives.

    ``module_attr`` names the physical linear on the owning block; every block the
    layout applies to (see ``ModuleLayout.present_when_attr``) must have it.
    ``adapter_attr`` fixes the adapter's registered name — it appears in
    checkpoint keys, so it is explicit rather than derived.
    """

    projection: ProjectionSpec
    module_attr: str
    in_dim: DimFn
    out_dim: DimFn
    adapter_attr: str
    guard: GuardFn | None = None
    adapter_class: type[LoRALinear] = LoRALinear


@dataclass(frozen=True)
class FusedAttach:
    """Several logical projections adapted on one fused physical linear.

    ``build`` constructs the family's split adapter (``LoRASplitQKV`` /
    ``LoRASplitFC1``) from the active subset; the fused host's geometry differs
    per family, so construction stays a callable while everything around it is
    shared.
    """

    module_attr: str
    projections: tuple[ProjectionSpec, ...]
    adapter_attr: str
    build: FusedBuildFn

    @property
    def targets(self) -> frozenset[str]:
        return frozenset(projection.hf_leaf for projection in self.projections)


@dataclass(frozen=True)
class ModuleLayout:
    """The complete adapter table for one block kind of one architecture.

    ``present_when_attr`` names the attribute that marks a block of this kind (a hybrid
    model's mixer layers lack it); the layout applies to every block that has it.
    ``full_sequence`` marks a block that gathers the sequence-parallel input itself and
    runs its linears without SP, so its adapters follow the non-SP TP semantics.
    """

    name: str
    fused: tuple[FusedAttach, ...] = ()
    singles: tuple[ProjectionBinding, ...] = ()
    present_when_attr: str | None = None
    hf_block_prefix: str | None = None
    full_sequence: bool = False

    @property
    def targets(self) -> frozenset[str]:
        names = [projection.hf_leaf for group in self.fused for projection in group.projections]
        names.extend(binding.projection.hf_leaf for binding in self.singles)
        assert len(names) == len(set(names)), f"layout {self.name!r} declares duplicate projection names"
        return frozenset(names)

    @property
    def fused_targets(self) -> frozenset[str]:
        """Names living in fused physical linears (union over this layout's groups)."""
        return frozenset(projection.hf_leaf for group in self.fused for projection in group.projections)


def layout_applies_to(block: nn.Module, layout: ModuleLayout) -> bool:
    return layout.present_when_attr is None or hasattr(
        block, layout.present_when_attr
    )  # config-access-exempt: the architecture layout names an optional block marker


def attach_layout(block: nn.Module, layout: ModuleLayout, layer: nn.Module, context: AttachContext) -> int:
    """Attach every targeted projection of ``layout`` to ``block`` inside ``layer``; return the adapter count."""
    if not layout_applies_to(block, layout):
        return 0
    if layout.full_sequence:
        context = dataclasses.replace(context, full_sequence=True)

    count = 0
    for group in layout.fused:
        host = getattr(
            block, group.module_attr
        )  # config-access-exempt: the architecture group names its host module attribute
        hf_prefix = context.naming.block_prefix(
            layer, host.weight, [projection.hf_leaf for projection in group.projections]
        )
        active = tuple(
            projection for projection in group.projections if context.lora.selects(f"{hf_prefix}{projection.hf_leaf}")
        )
        if not active:
            continue
        adapter = group.build(block, hf_prefix, context, active)
        setattr(block, group.adapter_attr, adapter)
        attach_adapter_forward(host, adapter, context.lora.scale)
        count += 1

    for binding in layout.singles:
        host = getattr(
            block, binding.module_attr
        )  # config-access-exempt: the architecture binding names its host module attribute
        hf_prefix = context.naming.block_prefix(layer, host.weight, [binding.projection.hf_leaf])
        if not context.lora.selects(f"{hf_prefix}{binding.projection.hf_leaf}"):
            continue
        out_features = binding.out_dim(block, context)
        if binding.guard is not None:
            binding.guard(host, context, binding.projection, out_features)
        adapter = binding.adapter_class(
            hf_prefix=hf_prefix,
            projection=binding.projection,
            reference=host.weight,
            context=context,
            in_features=binding.in_dim(block, context),
            out_features=out_features,
        )
        setattr(block, binding.adapter_attr, adapter)
        attach_adapter_forward(host, adapter, context.lora.scale)
        count += 1
    return count


class LayoutSpec:
    """Base class for shipped specs: declare ``layout`` once, everything derives.

    Subclasses set ``name`` and ``layout`` as class attributes; the supported
    target set and the attach walk come from the layout, so there is no
    parallel module-level frozenset to keep in sync. The Protocols in
    ``spec.base`` remain the external contract — custom providers do not have
    to inherit from this hierarchy.
    """

    name: str = ""
    layout: ModuleLayout = ModuleLayout(name="empty")

    @property
    def supported_targets(self) -> frozenset[str]:
        return self.layout.targets

    @property
    def canonical_targets_csv(self) -> str:
        """Targets in declaration order, as launchers pass them to --target-modules."""
        names = [projection.hf_leaf for group in self.layout.fused for projection in group.projections]
        names.extend(binding.projection.hf_leaf for binding in self.layout.singles)
        return ",".join(names)

    def applies_to(self, block: nn.Module) -> bool:
        return layout_applies_to(block, self.layout)

    def attach(self, block: nn.Module, layer: nn.Module, context: AttachContext) -> int:
        return attach_layout(block, self.layout, layer, context)


class AttentionSpecBase(LayoutSpec):
    """Base for attention specs: adds the family tag, HF block name, and validation hook."""

    family: AttentionFamily = AttentionFamily.GQA

    @property
    def hf_block(self) -> str:
        return (self.layout.hf_block_prefix or "self_attn.").rstrip(".")

    def validate(self, config, *, tp_size: int) -> None:
        del config, tp_size
